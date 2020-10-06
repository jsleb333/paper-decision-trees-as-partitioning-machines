import os
from importlib import reload

import partitioning_machines.utils.wedderburn_etherington_cache

def wedderburn_etherington(n):
    """
    Computes the Wedderburn Etherington numbers. They corresponds to the number of structurally non-equivalent binary trees with 'n' leaves.
    Args:
        n (int): The number of leaves of the binary tree.
    """
    cache = partitioning_machines.utils.wedderburn_etherington_cache.cache
    if n in cache:
        return cache[n]
    
    else:
        val = 0
        if n % 2 == 1:
            for i in range(1, n//2+1):
                val += wedderburn_etherington(i) * wedderburn_etherington(n-i)
        else:
            for i in range(1, n//2):
                val += wedderburn_etherington(i) * wedderburn_etherington(n-i)
            val += wedderburn_etherington(n//2)*(wedderburn_etherington(n//2) + 1)//2

        cache[n] = val
        
        with open(os.path.dirname(__file__) + '/wedderburn_etherington_cache.py', 'w') as file:
            file.write(f"cache = {cache}")
        reload(partitioning_machines.utils.wedderburn_etherington_cache)
        
        return val
