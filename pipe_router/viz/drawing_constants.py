"""
Visualization constants (colors, line types, etc.)
"""

DRAW_ARGS_VOLUME = {
    '_box_type': 'wireframe',
    'color': 'gray',
    'alpha': 0.5
}

DRAW_ARGS_GRID = {
    'color': 'gray',
    'alpha': 0.15
}

DRAW_ARGS_OCCUPIED_SPACE = {
    '_box_type': 'shaded_wireframe',
    'shaded': {
        'color': 'blue',
        'linestyle': '-',
        'alpha': 0.1
    },
    'wireframe': {
        'color': 'blue',
        'alpha': 0.3
    }
}

DRAW_ARGS_FREE_SPACE = {
    '_box_type': 'wireframe',
    'color': 'indigo',
    'linestyle': '--',
    'alpha': 0.15
}

DRAW_ARGS_CONNECTION = {
    'color': 'red',
    'alpha': 0.5,
    'linewidth': 5
}
