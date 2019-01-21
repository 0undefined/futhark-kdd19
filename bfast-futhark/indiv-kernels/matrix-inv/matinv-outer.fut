-- This is a version of code for gauss-jordan martrix inversion kernel
--    of bfast that uses only the outermost level of parallelism.
-- ==
-- compiled input @ ../../data/sahara.in.gz
-- compiled input @ ../../data/peru.in.gz
-- compiled input @ ../../data/d-16384-1024-512.in.gz
-- compiled input @ ../../data/d-16384-512-256.in.gz
-- compiled input @ ../../data/d-32768-256-128.in.gz
-- compiled input @ ../../data/d-32768-512-256.in.gz
-- compiled input @ ../../data/d-65536-128-64.in.gz
-- compiled input @ ../../data/d-65536-256-128.in.gz

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

let mat_inv [n] (A: [n][n]f32): [][]f32 =
    if (f32.isnan (A[0,0])) then A else -- COMPILER BUG!!!
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

entry main [m][k] (Xsqr: [m][k][k]f32) =
    map mat_inv Xsqr

