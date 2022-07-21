# pyLLTandVLM

# before running test do:
```
export NUMBA_DISABLE_JIT=1
```
[![CircleCI](https://circleci.com/gh/ggruszczynski/sailingVLM/tree/main.svg?style=svg&circle-token=c62058c5c0f125149f3e70358b3280a403d2b2b2)](https://circleci.com/gh/ggruszczynski/sailingVLM/tree/main)
python implementation of a 3D Vortex Lattice Method

```
            The geometry is described using the following CSYS.
            lem - leading edge (luff) of the main sail
            tem - trailing edge (leech) of the main sail

                        Z ^ (mast)     
                          |
                         /|
                        / |
                       /  |              ^ Y     
                   lem_NW +--+tem_NE    / 
                     /    |   \        /
                    /     |    \      /
                   /      |     \    /
                  /       |      \  /
                 /        |       \/
                /         |       /\
               /          |      /  \
              /           |     /    \
             /            |    /      \
            /      lem_SW |---/--------+tem_SE
           /              |  /          
  (bow) ------------------|-/-------------------------| (stern)
         \                |/                          |
    ------\---------------*---------------------------|-------------------------> X (water level)
        
```
