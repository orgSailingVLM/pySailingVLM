**NAJNOWSZA WERSJA** 

znajduje się na moim gicie w gałęzi main.

## uruchaminie kodu w trybie developerskim

Nie ma aktualnie już pliku requirements.txt, zamaist tego, by mieć wszystkie wymagane moduły należy wykonać poniższą komendę będąc w lokalizacji w której jest plik setup.cfg

```
~~~
pip install -q -e .
~~~

```

Dokona to lokalnej instalacji i zainstalują się wykagane paczki (patrz plik setup.cfg 'install_requires')

## zbudowanie paczki lokalnie

Jeżeli jest to wymagane:

```
~~~
pip install --upgrade build
~~~
~~~
pip install build
~~~
Następnie będąc w lokalizacji w której znajduje się plik setup.cfg
~~~
python3 -m build
pip install dist/pySailingVLM-1.0.0.tar.gz
~~~
```

**UWAGA!!!**

Pierwsze użycie może pokazywać warningi z numby, tak ma być. 
Powtórne użycie nie spowoduje już warningów. Numba musi sobie zcache'ować to i owo.

```
pySailingVLM --help

Output:

/home/zuzanna/Documents/miniconda3/envs/sv_build_test_2/lib/python3.10/site-packages/numba/core/lowering.py:107: NumbaDebugInfoWarning: Could not find source for function: <function __numba_array_expr_0x7fbf333f1780 at 0x7fbf33361b40>. Debug line information may be inaccurate.
  warnings.warn(NumbaDebugInfoWarning(msg))
/home/zuzanna/Documents/miniconda3/envs/sv_build_test_2/lib/python3.10/site-packages/numba/core/lowering.py:107: NumbaDebugInfoWarning: Could not find source for function: <function __numba_array_expr_0x7fbf333f16c0 at 0x7fbf3390bf40>. Debug line information may be inaccurate.

```

## uruchomienie skryptu z cli

Aby uruchomić program z command line należy przygotować plik o nazwie varaibles.py z zawartościa taką jak np. sweep.py w pySailingVLM/examples/sweep.py na gicie
[przyklad](https://github.com/zuza3012/sailingVLM/blob/main/pySailingVLM/examples/sweep.py).
Plik powinien znajdować się w working directory lub w innej, sprecyzowanej przez odpowiedni przełącznik (zobacz pySailingVLM --help).

Wynikiem takiego użycia jest pojawienie się okenka z rysunkiem żaglówki w matplotlibie, zapisanie plików wynikowych (xlsx plus variables.py) oraz wykres 
przedstawiającym rozkład współczynników ciśnienia na zaglach (patrz dołączony rysunek)

## Uruchomienie z pythona

~~~
from sailing_vlm.runner import aircraft as a
import numpy as np
chord_length = 1.0             
half_wing_span = 5.0 
AoA_deg = 10.
sweep_angle_deg = 0.
tws_ref = 1.0
V = np.array([tws_ref, 0., 0.])
rho = 1.225
gamma_orientation = -1.0
n_spanwise = 32  # No of control points (above the water) per sail, recommended: 50
n_chordwise = 8 #?
b = a.Aircraft(chord_length, half_wing_span, AoA_deg, n_spanwise, n_chordwise, V, rho, gamma_orientation)
b.Cxyz
>>> (0.023472780216173314, 0.8546846987984326, 0.0) 
~~~

## jupyter notebook

Patrz dołącznoy plik notebooka.

<https://janakiev.com/blog/jupyter-virtual-envs/>

Jupyter Notebook makes sure that the IPython kernel is available, 
but you have to manually add a kernel with a different version of Python or a virtual environment.

```.sh
pip install --user ipykernel
#Next you can add your virtual environment to Jupyter by typing:
python -m ipykernel install --user --name=myenv
```
