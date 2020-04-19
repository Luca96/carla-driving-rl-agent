## Instructions
Before running any code from this repo you have to:
1. **Install CARLA Simulator 0.9.8**: [carla.readthedocs.io/en/latest/start_quickstart/](https://carla.readthedocs.io/en/latest/start_quickstart/)
2. **Install CARLA's Python bindings**: 
    * `cd your-path-to-carla/CARLA_0.9.8/PythonAPI/carla/dist/`
    * Extract `carla-0.9.8-py3.5-XXX-x86_64.egg` where `XXX` depends on your OS, i.e. `linux` or `windows`
    * Create a `setup.py` file within the extracted folder and write the following:
      ```python
      from distutils.core import setup
      
      setup(name='carla',
            version='0.9.8',
            py_modules=['carla']) 
      ```
    * Install via pip: `pip install -e ~/CARLA_0.9.8/PythonAPI/carla/dist/carla-0.9.8-py3.5-YOUR_OS-x86_64`
3. **Clone this repo**
4. **Run the CARLA Simulator**: `your-path-to/CARLA_0.9.8/./CarlaUE4.sh`
    * To use less resources add these flags: `-windowed -ResX=8 -ResY=8 --quality-level=Low`
5. Enjoy!

## Examples
...
