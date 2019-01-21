-- A "fused" version of bfast that; the creation of "X" and the batched matrix-matrix
-- multiplication are kept separate (otherwise performance is awaful), the remaining
-- steps up to and including the calculation of "hmax" is fused in one kernel.
-- ==
-- compiled input @ data/sahara.in.gz
-- compiled input @ data/peru.in.gz
-- compiled input @ data/d-16384-1024-512.in.gz
-- compiled input @ data/d-16384-512-256.in.gz
-- compiled input @ data/d-32768-256-128.in.gz
-- compiled input @ data/d-32768-512-256.in.gz
-- compiled input @ data/d-65536-128-64.in.gz
-- compiled input @ data/d-65536-256-128.in.gz

-- output @ data/sahara.out.gz


let logplus (x: f32) : f32 =
  if x > (f32.exp 1)
  then f32.log x else 1

let adjustValInds [N] (n : i32) (ns : i32) (Ns : i32) (val_inds : [N]i32) (ind: i32) : i32 =
    if ind < Ns - ns then (unsafe val_inds[ind+ns]) - n else -1

let filterPadWithKeys [n] 't
           (p : (t -> bool))
           (dummy : t)           
           (arr : [n]t) : ([n](t,i32), i32) =
  let tfs = map (\a -> if p a then 1 else 0) arr
  let isT = scan (+) 0 tfs
  let i   = last isT
  let inds= map2 (\a iT -> if p a then iT-1 else -1) arr isT
  let rs  = scatter (replicate n dummy) inds arr
  let ks  = scatter (replicate n 0) inds (iota n)
  in  (zip rs ks, i) 

-- | builds the X matrices; first result dimensions of size 2*k+2
let mkX_with_trend [N] (k2p2: i32) (f: f32) (mappingindices: [N]i32): [k2p2][N]f32 =
  map (\ i ->
        map (\ind ->
                if i == 0 then 1f32
                else if i == 1 then r32 ind
                else let (i', j') = (r32 (i / 2), r32 ind)
                     let angle = 2f32 * f32.pi * i' * j' / f 
                     in  if i % 2 == 0 then f32.sin angle 
                                       else f32.cos angle
            ) mappingindices
      ) (iota k2p2)

let mkX_no_trend [N] (k2p2m1: i32) (f: f32) (mappingindices: [N]i32): [k2p2m1][N]f32 =
  map (\ i ->
        map (\ind ->
                if i == 0 then 1f32
                else let i = i + 1
		     let (i', j') = (r32 (i / 2), r32 ind)
                     let angle = 2f32 * f32.pi * i' * j' / f 
                     in  if i % 2 == 0 then f32.sin angle 
                                       else f32.cos angle
            ) mappingindices
      ) (iota k2p2m1)

---------------------------------------------------
-- Adapted matrix inversion so that it goes well --
-- with intra-blockparallelism                   --
---------------------------------------------------

  let gauss_jordan [nm] (n:i32) (m:i32) (A: *[nm]f32): [nm]f32 =
    loop A for i < n do
      let v1 = A[i]
      let A' = map (\ind -> let (k, j) = (ind / m, ind % m)
                            in if v1 == 0.0 then unsafe A[k*m+j] else
                            let x = unsafe (A[j] / v1) in
                                if k < n-1  -- Ap case
                                then unsafe ( A[(k+1)*m+j] - A[(k+1)*m+i] * x )
                                else x      -- irow case
                   ) (iota nm)
      in  scatter A (iota nm) A'

  let mat_inv [n] (A: [n][n]f32): [n][n]f32 =
    let m = 2*n
    let nm= n*m
    -- Pad the matrix with the identity matrix.
    let Ap = map (\ind -> let (i, j) = (ind / m, ind % m)
                          in  if j < n then unsafe ( A[i,j] )
                                       else if j == n+i
                                            then 1.0
                                            else 0.0
                 ) (iota nm)
    let Ap' = gauss_jordan n m Ap
    -- Drop the identity matrix at the front!
    in (unflatten n m Ap')[0:n,n:2*n]
--------------------------------------------------
--------------------------------------------------

let dotprod [n] (xs: [n]f32) (ys: [n]f32): f32 =
  reduce (+) 0.0 <| map2 (*) xs ys

let matvecmul_row [n][m] (xss: [n][m]f32) (ys: [m]f32) =
  map (dotprod ys) xss

let dotprod_filt [n] (vct: [n]f32) (xs: [n]f32) (ys: [n]f32) : f32 =
  f32.sum (map3 (\v x y -> x * y * if (f32.isnan v) then 0.0 else 1.0) vct xs ys)

let matvecmul_row_filt [n][m] (xss: [n][m]f32) (ys: [m]f32) =
    map (\xs -> map2 (\x y -> if (f32.isnan y) then 0 else x*y) xs ys |> f32.sum) xss

let matmul_filt [n][p][m] (xss: [n][p]f32) (yss: [p][m]f32) (vct: [p]f32) : [n][m]f32 =
  map (\xs -> map (dotprod_filt vct xs) (transpose yss)) xss

----------------------------------------------------
----------------------------------------------------

-- | implementation is in this entry point
--   the outer map is distributed directly
entry main [m][N] (trend: i32) (k: i32) (n: i32) (freq: f32)
                  (hfrac: f32) (lam: f32)
                  (mappingindices : [N]i32)
                  (images : [m][N]f32) =
  ----------------------------------
  -- 1. make interpolation matrix --
  ----------------------------------
  let k2p2 = 2*k + 2
  let k2p2' = if trend > 0 then k2p2 else k2p2-1
  let X = intrinsics.opaque <|
          if trend > 0
              then mkX_with_trend k2p2' freq mappingindices
          else mkX_no_trend   k2p2' freq mappingindices

  -- PERFORMANCE BUG: instead of `let Xt = copy (transpose X)`
  --   we need to write the following ugly thing to force manifestation:
  let zero = r32 <| (N*N + 2*N + 1) / (N + 1) - N - 1
  let Xt  = map (map (+zero)) (copy (transpose X))
            |> intrinsics.opaque
  let Xh  =  (X[:,:n])
  let Xth =  (Xt[:n,:])
      
  let (tup1s, tup2s) = unzip <|
    map (\Y ->
      let Yh  = Y[:n]
      ----------------------------------
      -- 2. mat-mat multiplication    --
      ----------------------------------
      let Xsqr = matmul_filt Xh Xth Yh

      ----------------------------------
      -- 3. matrix inversion          --
      ----------------------------------
      let Xinv = mat_inv Xsqr
      ---------------------------------------------
      -- 4. several matrix-vector multiplication --
      ---------------------------------------------
      let beta0  = matvecmul_row_filt Xh Yh   -- [2k+2]

      let beta   = matvecmul_row Xinv beta0    -- [2k+2]

      let y_pred = matvecmul_row Xt   beta     -- [N]

      ---------------------------------------------
      -- 5. filter etc.                          --
      ---------------------------------------------
      let y_error_all = map2 (\ye yep -> if !(f32.isnan ye) 
                                         then ye-yep else f32.nan
                             ) Y y_pred
      let (tups, Ns) = filterPadWithKeys (\y -> !(f32.isnan y)) (f32.nan) y_error_all
      let (y_error, val_inds) = unzip tups

      ---------------------------------------------
      -- 6. ns and sigma                         --
      ---------------------------------------------
      let ns    = map (\ye -> if !(f32.isnan ye) then 1 else 0) Yh
                  |> reduce (+) 0
      let sigma = map (\i -> if i < ns then unsafe y_error[i] else 0.0) (iota n)
                  |> map (\ a -> a*a ) |> reduce (+) 0.0
      let sigma = f32.sqrt ( sigma / (r32 (ns-k2p2)) )
      let h     = t32 ( (r32 ns) * hfrac )
      in  ((h, Ns, ns), (sigma, y_error, val_inds))
    ) images
  in
  let (hs, Nss, nss) = unzip3 tup1s
  let (sigmas, y_errors, val_indss) = unzip3 tup2s
  ---------------------------------------------
  -- 7. moving sums first and bounds:        --
  ---------------------------------------------
  let hmax = reduce_comm (i32.max) 0 hs

  let BOUND = map (\q -> let t   = n+1+q
                         let time = unsafe mappingindices[t-1]
                         let tmp = logplus ((r32 time) / (r32 mappingindices[N-1]))
                         in  lam * (f32.sqrt tmp)
                  ) (iota (N-n))

  ---------------------------------------------
  -- 8. moving sums computation:             --
  ---------------------------------------------
  let (_MOs, _MOs_NN, breaks, means) = zip (zip4 Nss nss sigmas hs) (zip y_errors val_indss) |>
    map (\ ( (Ns,ns,sigma, h), (y_error,val_inds) ) ->
            let MO_fst = map (\i -> if i < h then unsafe y_error[i + ns-h+1] else 0.0) (iota hmax)
                         |> reduce (+) 0.0 
            let Nmn = N-n
            let MO = map (\j -> if j >= Ns-ns then 0.0
                                else if j == 0 then MO_fst
                                else unsafe (-y_error[ns-h+j] + y_error[ns+j])
                         ) (iota Nmn) |> scan (+) 0.0
	        
            let MO' = map (\mo -> mo / (sigma * (f32.sqrt (r32 ns))) ) MO
	        let (is_break, fst_break) = 
		        map3 (\mo b j ->  if j < Ns - ns && !(f32.isnan mo)
			                      then ( (f32.abs mo) > b, j )
                                  else ( false, j )
		             ) MO' BOUND (iota Nmn)
		        |> reduce_comm (\ (b1,i1) (b2,i2) -> 
                                  if b1 then (b1,i1) 
                                  else if b2 then (b2, i2)
                                  else (b1,i1) 
                  	      	   ) (false, -1)
            let mean = map2 (\x j -> if j < Ns - ns then x else 0.0 ) MO' (iota Nmn)
                       |> reduce (+) 0.0

            let fst_break' = if !is_break then -1
                             else let adj_break = adjustValInds n ns Ns val_inds fst_break
                                  in  ((adj_break-1) / 2) * 2 + 1  -- Cosmin's validation hack
            let fst_break' = if ns <=5 || Ns-ns <= 5 then -2 else fst_break'

            let val_inds' = map (adjustValInds n ns Ns val_inds) (iota Nmn)
            let MO'' = scatter (replicate Nmn f32.nan) val_inds' MO'
            in (MO'', MO', fst_break', mean)
        ) |> unzip4

  in (breaks, means)

