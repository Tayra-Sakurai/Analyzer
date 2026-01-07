import os
import analysismodule
import numpy as np
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilenames

SCHEMA: analysismodule.SchemaType = {
    "id": 0,
    "position": (1, 2, 3)
}

files = askopenfilenames(
    title='File Picker',
    filetypes=(
        (
            'LAMMPS Trajectory',
            '*.lammpstrj'
        ),
    ),
    initialdir=os.curdir,
    defaultextension='.lammpstrj'
)

if files == '':
    raise ValueError('Unexpected Value!')

analysis = analysismodule.ProcessAnalysis(
    files,
    SCHEMA
)

print('\\begin{longtblr}[]{}')
print('    \\toprule')
for i in range(1, 151):
    a, err = analysis.regress(i)
    print(f'    {i} & {a:.2f}({round(err * 100)}) & {a / 2:.2f}({round(err * 50)})\\\\')
print('    \\bottomrule')
print('\\end{longtblr}')

points = analysis.get_points(150)
print(np.log10(points[0] - 1))
print(np.log10(points[1]))
print(np.log10(np.e)/points[1])
print()
print(points[0] - 1)
print(points[1])
a1, _ = analysis.regress(150)
print()
print()
x = np.array((10., 20.))
y = 10 ** (a * np.log10(x))
print(x)
print(y)
