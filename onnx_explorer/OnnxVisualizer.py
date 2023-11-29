import onnx
import onnx.tools.net_drawer as net_drawer
from graphviz import Digraph

class OnnxVisualizer:
    def __init__(self, model_path):
        self.model = onnx.load(model_path)

    def visualize_model(self, output_path):
        graph = net_drawer.GetPydotGraph(self.model.graph, name=self.model.graph.name, rankdir="TB",
                                         node_producer=self._node_producer)
        graph.write_png(output_path)

    def _node_producer(self, node, **kwargs):
        return net_drawer.Node(node.name, label=node.name, shape="box", **kwargs)

    def _edge_producer(self, edge, **kwargs):
        return net_drawer.Edge(edge.src, edge.dst, label=edge.src, **kwargs)

if __name__ == '__main__':
    model_path = "/Users/gatilin/youtu-work/SVAP/modified_det_model_float32.onnx"
    visualizer = OnnxVisualizer(model_path)
    visualizer.visualize_model("model_structure.png")