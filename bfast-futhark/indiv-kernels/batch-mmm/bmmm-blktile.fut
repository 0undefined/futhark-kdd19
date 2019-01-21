-- This is a block-tiled version of code for batched matrix-matrix
--   multiplication kernel of bfast, i.e., it only uses block tiling.
--   It works directly with bfast's datasets.
-- REMEMBER to run with --default-tile-size=8, as in:
--   $ futhark bench --backend opencl --pass-option --default-tile-size=8 bmmm-blktile.fut
-- ==
-- compiled input @ ../../data/sahara.in.gz
-- compiled input @ ../../data/peru.in.gz
-- compiled input @ ../../data/d-16384-1024-512.in.gz
-- compiled input @ ../../data/d-16384-512-256.in.gz
-- compiled input @ ../../data/d-32768-256-128.in.gz
-- compiled input @ ../../data/d-32768-512-256.in.gz
-- compiled input @ ../../data/d-65536-128-64.in.gz
-- compiled input @ ../../data/d-65536-256-128.in.gz

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

-- | dot-product but in which we filter-out the entries for which `vct[i]==NAN`
let dotprod_filt [n] (vct: [n]f32) (xs: [n]f32) (ys: [n]f32) : f32 =
  f32.sum (map3 (\i x y -> x * y * if (f32.isnan (vct[i])) then 0.0 else 1.0) (iota n) xs ys)

-- | matrix-matrix multiplication but with NAN-filtering on `vct`
let matmul_filt [n][p][m] (xss: [n][p]f32) (yss: [p][m]f32) (vct: [p]f32) : [n][m]f32 =
  map (\xs -> map (dotprod_filt vct xs) (transpose yss)) xss

-- | implementation is in this entry point
--   the outer map is distributed directly
entry main [m][N] (_trend: i32) (k: i32) (n: i32) (freq: f32)
                  (_hfrac: f32) (_lam: f32)
                  (mappingindices : [N]i32)
                  (images : [m][N]f32) =
  ----------------------------------
  -- 1. make interpolation matrix --
  ----------------------------------
  let k2p2 = 2*k + 2
  let X = mkX_with_trend k2p2 freq (mappingindices[:n])
  let Xt= copy (transpose X)
  
  let Yh  =  (images[:,:n])
  
  let Xsqr = map (matmul_filt X Xt) Yh
  in  Xsqr
