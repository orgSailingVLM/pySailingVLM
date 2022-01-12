# pyLLTandVLM

[![CircleCI](https://circleci.com/gh/ggruszczynski/sailingVLM/tree/main.svg?style=svg&circle-token=fd8847cf98210cd3e3811b82c4ab8f639d50dd55)](https://circleci.com/gh/ggruszczynski/sailingVLM/tree/main)

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
