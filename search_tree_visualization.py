from abc import ABC, abstractmethod
from typing import Literal
from aostar_data_structures import NodeDetailedState, Node, ANDNode, ORNode

class Markers(ABC):
    @abstractmethod
    def paint(self, text: str, color: Literal[
        'red', 'blue', 'green', 'yellow', 'cyan'
    ]) -> str:
        pass
    @abstractmethod
    def emphasize(self, text: str) -> str:
        pass
    @abstractmethod
    def downplay(self, text: str) -> str:
        pass

    @property
    @abstractmethod
    def prologue(sef) -> str:
        pass
    @property
    @abstractmethod
    def epilogue(sef) -> str:
        pass


class ANSIMarkers(Markers):
    def paint(self, text: str, color: Literal[
        'red', 'blue', 'green', 'yellow', 'cyan'
    ]) -> str:
        match color:
            case 'red':
                return '\033[91m' + text + '\033[0m'
            case 'blue':
                return '\033[94m' + text + '\033[0m'
            case 'green':
                return '\033[92m' + text + '\033[0m'
            case 'yellow':
                return '\033[93m' + text + '\033[0m'
            case 'cyan':
                return '\033[96m' + text + '\033[0m'
    def emphasize(self, text: str) -> str:
        return '\033[1m' + text + '\033[0m'
    def downplay(self, text: str) -> str:
        return '\033[2m' + text + '\033[0m'

    @property
    def prologue(self) -> str:
        paint = self.paint
        downplay = self.downplay
        return f"""
Legend.
  "{        (paint('├──', 'green'))}": solved AND node that's part of the solution
  "{        (paint('├──', 'blue' ))}": solved OR node that's part of the solution
  "{downplay(paint('├──', 'green'))}": solved AND node that's not part of the solution
  "{downplay(paint('├──', 'blue' ))}": solved AND node that's not part of the solution
  "{        (     ('├──'         ))}": active node that hasn't been solved and is pending for more work
  "{        (     ('├✖─'         ))}": failed node whose all children failed, or failed AND node whose tactic doesn't compile
  "{        (     ('├↺─'         ))}": failed AND node whose tactic makes no progress (e.g. leading to the same goal)
  "{downplay(     ('├──'         ))}": failed AND node whose tactic is repeatedly suggested

"""
    @property
    def epilogue(self) -> str:
        return ''

class HTMLMarkers(Markers):
    def paint(self, text: str, color: Literal[
        'red', 'blue', 'green', 'yellow', 'cyan'
    ]) -> str:
        match color:
            case 'red':
                return '<span style="color:red;">'    + text + '</span>'
            case 'blue':
                return '<span style="color:blue;">'   + text + '</span>'
            case 'green':
                return '<span style="color:green;">'  + text + '</span>'
            case 'yellow':
                return '<span style="color:yellow;">' + text + '</span>'
            case 'cyan':
                return '<span style="color:cyan;">'   + text + '</span>'
    def emphasize(self, text: str) -> str:
        return '<strong>' + text + '</strong>'
    def downplay(self, text: str) -> str:
        return '<span style="opacity:0.5;">' + text + '</span>'

    @property
    def prologue(self) -> str:
        paint = self.paint
        downplay = self.downplay
        return """
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Zoom Control Example</title>
    <style>
        body {
            transition: transform 0.2s; /* Smooth transition for zooming */
            transform-origin: top left; /* Set the origin for scaling */
        }
        #zoomableContent {
            transition: transform 0.2s; /* Smooth transition for zooming */
            transform-origin: top left; /* Set the origin for scaling */
        }
        #zoomSlider {
            margin: 20px;
            width: 300px; /* Width of the slider */
        }
    </style>
</head>
<body><code>
    <label for="zoomSlider">Zoom Level:</label>
    <input type="range" id="zoomSlider" min="0.1" max="1" step="0.1" value="1">
""" +\
f"""
    <p>Legend:</p>
    <ul>
    <li>"{        (paint('├──', 'green'))}": solved AND node that's part of the solution                                            </li>
    <li>"{        (paint('├──', 'blue' ))}": solved OR node that\'s part of the solution                                            </li>
    <li>"{downplay(paint('├──', 'green'))}": solved AND node that\'s not part of the solution                                       </li>
    <li>"{downplay(paint('├──', 'blue' ))}": solved OR node that\'s not part of the solution                                        </li>
    <li>"{        (     ('├──'         ))}": active node that hasn\'t been solved and is pending for more work                      </li>
    <li>"{        (     ('├✖─'         ))}": failed node whose all children failed, or failed AND node whose tactic doesn't compile </li>
    <li>"{        (     ('├↺─'         ))}": failed AND node whose tactic makes no progress (e.g. leading to the same goal)         </li>
    <li>"{downplay(     ('├──'         ))}": failed AND node whose tactic is repeatedly suggested                                   </li>
    </ul>

    <div id="zoomableContent">
    <pre><code>
"""
    @property
    def epilogue(self) -> str:
        return """
    </code></pre>
    </div>
    
    <script>
        const zoomSlider = document.getElementById('zoomSlider');
        const zoomableContent = document.getElementById('zoomableContent');

        // Function to update the zoom level
        function updateZoom() {
            const zoomLevel = zoomSlider.value;
            zoomableContent.style.transform = `scale(${zoomLevel})`;
            zoomableContent.style.width = `${100 / zoomLevel}%`; // Adjust width to prevent clipping
        }

        // Event listener for the slider
        zoomSlider.addEventListener('input', updateZoom);
    </script><br>
</body>
"""


class PlainTextMarkers(Markers):
    def paint(self, text: str, color: Literal[
        'red', 'blue', 'green', 'yellow', 'cyan'
    ]) -> str:
        return ''
    def emphasize(self, text: str) -> str:
        return ''
    def downplay(self, text: str) -> str:
        return ''

    @property
    def prologue(self) -> str:
        return '' # No legend to show
    @property
    def epilogue(self) -> str:
        return ''


def present_search_tree(
    node: Node,
    style: Literal['ANSI', 'HTML', 'plain'] = 'plain',
    is_part_of_solution: bool = True,
    prefix: str = '',
    is_last: bool = True
) -> str:
    # Prints the search tree in a format similar to the Linux `tree` command.
    # Parts that are solved are in green or blue or boldface
    # Others, black and non-bold
    assert not node.hide_from_visualization, "Attempting to print a node that should be hidden from visualization."
    is_part_of_solution = is_part_of_solution and node.solved

    match style:
        case 'ANSI':
            markers = ANSIMarkers()
        case 'HTML':
            markers = HTMLMarkers()
        case 'plain':
            markers = PlainTextMarkers()
        case _:
            raise ValueError(f"Unexpected style {style}")
    paint = markers.paint
    match node:
        case ANDNode(_):
            endorse = lambda text: paint(text, 'green')
        case ORNode(_):
            endorse = lambda text: paint(text, 'blue')
        case _:
            # MERISTEM node: should always be hidden
            raise TypeError(f"Unexpected node type {type(node)}")
    emphasize = markers.emphasize
    downplay = markers.downplay

    if not prefix:
        search_tree_str = markers.prologue
    else:
        search_tree_str = prefix
    sub_tree_prefix = prefix

    connector = '└' if is_last else '├'
    # '└── ' versus '├── ' for most cases

    match node.detailed_state:
        case NodeDetailedState.SOLVED if is_part_of_solution:
            search_tree_str +=          ( endorse( connector + '── '              ) )
            search_tree_str += emphasize(        ( str(node).replace("\n", "\\n") ) )
            sub_tree_prefix +=          ( endorse( '    ' if is_last else '│   '  ) )
        case NodeDetailedState.SOLVED if not is_part_of_solution:
            search_tree_str +=  downplay( endorse( connector + '── '              ) )
            search_tree_str +=          (        ( str(node).replace("\n", "\\n") ) )
            sub_tree_prefix +=  downplay( endorse( '    ' if is_last else '│   '  ) )
        case NodeDetailedState.ACTIVE:
            search_tree_str +=          (        ( connector + '── '              ) )
            search_tree_str +=          (        ( str(node).replace("\n", "\\n") ) )
            sub_tree_prefix +=          (        ( '    ' if is_last else '│   '  ) )
        case NodeDetailedState.FAILED_DUE_TO_CHILDREN | NodeDetailedState.DOESNT_COMPILE:
            search_tree_str +=          (        ( connector + '✖─ '              ) )
            search_tree_str +=          (        ( str(node).replace("\n", "\\n") ) )
            sub_tree_prefix +=          (        ( '    ' if is_last else '│   '  ) )
        case NodeDetailedState.NO_PROGRESS:
            search_tree_str +=          (        ( connector + '↺─ '              ) )
            search_tree_str +=          (        ( str(node).replace("\n", "\\n") ) )
            sub_tree_prefix +=          (        ( '    ' if is_last else '│   '  ) )
        case NodeDetailedState.ABANDONED | NodeDetailedState.IS_REPETITIVE:
            search_tree_str +=  downplay(        ( connector + '── '              ) )
            search_tree_str +=  downplay(        ( str(node).replace("\n", "\\n") ) )
            sub_tree_prefix +=  downplay(        ( '    ' if is_last else '│   '  ) )
    search_tree_str += '\n'

    visible_children = [child for child in node.children if not child.hide_from_visualization]
    for i, child in enumerate(visible_children):
        is_last_child = (i == len(visible_children) - 1)
        search_tree_str += present_search_tree(
            child,
            style,
            is_part_of_solution,
            sub_tree_prefix,
            is_last_child
        )
    
    if not prefix:
        search_tree_str += markers.epilogue
    
    return search_tree_str
