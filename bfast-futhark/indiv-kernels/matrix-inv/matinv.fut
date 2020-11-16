-- This is a intra-group-paralle version of code for gauss-jordan martrix
--   inversion kernel of bfast.
-- ==
-- compiled input @ ../../data/D1-Xsqr.in.gz
-- compiled input @ ../../data/D3-Xsqr.in.gz
-- compiled input @ ../../data/D5-Xsqr.in.gz
-- compiled input @ ../../data/peru-Xsqr.in.gz
-- compiled input @ ../../data/africa-Xsqr.in.gz

-- compiled input @ ../../data/sahara-Xsqr.in.gz

let gauss_jordan [nm] (n:i32) (m:i32) (A: *[nm]f32): [nm]f32 =
  loop A for i < n do
      let v1 = #[unsafe] A[i64.i32 i]
      let A' = map (\ind -> let (k, j) = (ind / m, ind % m)
                            in if v1 == 0.0 then #[unsafe] A[i64.i32(k*m+j)] else
                            let x = #[unsafe] (A[i64.i32(j)] / v1) in
                                if k < n-1  -- Ap case
                                then #[unsafe] ( A[i64.i32((k+1)*m+j)] - A[i64.i32((k+1)*m+i)] * x )
                                else x      -- irow case
                   ) (map i32.i64 (iota nm))
      in  scatter A (iota nm) A'

let mat_inv [n0] (A: [n0][n0]f32): [n0][n0]f32 =
    let n = i32.i64 n0
    let m = 2*n
    let nm= n*m
    -- Pad the matrix with the identity matrix.
    let Ap = map (\ind -> let (i, j) = (ind / m, ind % m)
                          in  if j < n then #[unsafe] ( A[i,j] )
                                       else if j == n+i
                                            then 1.0
                                            else 0.0
                 ) (map i32.i64 (iota (i64.i32 nm)))
    let Ap' = gauss_jordan n m Ap
    -- Drop the identity matrix at the front!
    in (unflatten (i64.i32 n) (i64.i32 m) Ap')[0:(i64.i32 n),(i64.i32 n): i64.i32 (2*n)] :> [n0][n0]f32

entry main [M][K] (Xsqr: [M][K][K]f32) =
    map mat_inv Xsqr

-- For the intra-group parallel version:
--   $ FUTHARK_INCREMENTAL_FLATTENING=1 futhark bench --backend opencl --pass-option --size=main.suff_intra_par_1=1 matinv.fut
-- For the flat version in global memory, run it simply with moderate flattening:
--   $ futhark bench --backend opencl matinv.fut
