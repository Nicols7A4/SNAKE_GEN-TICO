import numpy as np

class Cerebro:
    def __init__(self, n_inputs, n_outputs, genes=None):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        
        if genes is None:
            self.genes = np.random.uniform(-1, 1, (n_inputs, n_outputs))
        else:
            self.genes = genes

    def predecir(self, inputs):
        decision = np.dot(inputs, self.genes)
        return decision

    # --- NUEVOS MÉTODOS ---
    def guardar(self, filename):
        """Guarda los pesos en un archivo de texto .txt"""
        np.savetxt(filename, self.genes, fmt='%.5f')

    @staticmethod
    def cargar(filename):
        """Carga los pesos desde un archivo y devuelve un Cerebro nuevo"""
        genes_cargados = np.loadtxt(filename)
        # Inferimos inputs/outputs por la forma de la matriz
        inputs, outputs = genes_cargados.shape
        return Cerebro(inputs, outputs, genes_cargados)