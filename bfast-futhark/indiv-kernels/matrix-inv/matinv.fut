-- This is a intra-group-paralle version of code for gauss-jordan martrix
--   inversion kernel of bfast.
-- ==
-- compiled input @ ../../data/D1-Xsqr.in.gz
-- compiled input @ ../../data/D3-Xsqr.in.gz
-- compiled input @ ../../data/D5-Xsqr.in.gz
-- compiled input @ ../../data/peru-Xsqr.in.gz
-- compiled input @ ../../data/sahara-Xsqr.in.gz

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

entry main [M][K] (Xsqr: [M][K][K]f32) =
    map mat_inv Xsqr

-- For the intra-group parallel version:
--   $ FUTHARK_INCREMENTAL_FLATTENING=1 futhark bench --backend opencl --pass-option --size=main.suff_intra_par_1=1 matinv.fut
-- For the flat version in global memory, run it simply with moderate flattening:
--   $ futhark bench --backend opencl matinv.fut
