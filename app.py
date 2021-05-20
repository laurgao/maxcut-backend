import pennylane as qml
from pennylane import numpy as np
import networkx as nx
from scipy.optimize import minimize

np.random.seed(42)

from flask import Flask, json, jsonify, request, Response, stream_with_context

app = Flask(__name__) # __name__ references this file

@app.route('/qaoa', methods=['POST'])
def calculate_maxcut():
    data = request.get_json()
    edges = data["edges"]
    def generate():
        yield "Sorry, this takes a while... "
        # ALL USER INPUTS ## 
        # convert strings to numbers
        for i in range(len(edges)):
            edges[i][0] = int(edges[i][0])
            edges[i][1] = int(edges[i][1])
            edges[i][2] = float(edges[i][2])
        
        
        ## ADVANCED USER INPUTS ##
        # For those familiar with the algorithm - if you want to tweak any hyperparameters ;)

        num_layers = 4 # The number of layers to repeat our cost and mixer unitaries
        num_reps = 50 # The number of repetitions of the circuit when sampling probability distribution. AKA the number of shots.
        num_iters = 25 # The number of iterations our optimizer will go through when optimizing parameters
        init_params = 0.01 * np.random.rand(2, num_layers) # Initialize the parameters near zero. Generates array size 2, 4.

        nodes = []
        for edge in edges:
            start_node = edge[0]
            end_node = edge[1]
            if (start_node not in nodes):
                nodes.append(start_node)
            if (end_node not in nodes):
                nodes.append(end_node)
        num_nodes = len(nodes)

        graph = nx.Graph() 
        graph.add_nodes_from(nodes)
        graph.add_weighted_edges_from(edges)

        # Mixer layer with parameter beta
        def mixer_layer(beta):
            for wire in range(num_nodes):
                qml.RX(2 * beta, wires=wire)

        # Cost layer with parameter gamma
        def cost_layer(gamma):
            for edge in edges:
                wire1 = edge[0]
                wire2 = edge[1]
                weight = edge[2]
                qml.CNOT(wires=[wire1, wire2])
                qml.RZ(gamma*weight, wires=wire2) # Multiply gamma by the weight - this is the first algorithmetic change from the unweighted maxcut code
                qml.CNOT(wires=[wire1, wire2])

        def comp_basis_measurement(wires):
            num_nodes = len(wires)
            return qml.Hermitian(np.diag(range(2 ** num_nodes)), wires=wires)


        dev = qml.device("default.qubit", wires=num_nodes, shots=1)

        @qml.qnode(dev)
        def circuit(gammas, betas, edge=None, num_layers=1):
            
            # Applies a Hadamard gate to each qubit, which puts our circuit into the quantum state |+...+>
            # In this state, the probability of measuring any computational basis state is equal. Algorithms are commonly initialized with all states in equal superposition.
            for wire in range(num_nodes):
                qml.Hadamard(wires=wire)
                
            # Repeat the cost and mixer layers p times each
            for layer in range(num_layers):
                cost_layer(gammas[layer])
                mixer_layer(betas[layer])
                
            # Take the measurement of all qubits in the computational basis
            measurement = qml.sample(comp_basis_measurement(range(num_nodes)))
            return measurement

        
        def decimal_to_binary(decimal): # future: abstract this function to take in length.
            binary_num = []
            
            # Outputs bitstring of 1s and 0s into an array of digits
            def convert(decimal):
                if decimal >= 1:
                    convert(decimal // 2)
                    binary_num.append(decimal % 2)
            
            convert(decimal)
                
            # Change the binary number to have 4 digits, if it doesn't already
            for i in range(num_nodes + 1):
                if len(binary_num) < i:
                    binary_num.insert(0, 0) # At beginning append 0
            
            return binary_num # Outputs array of the digits of the binary number

        def get_counts(params):   
            gammas = [params[0], params[2], params[4], params[6]]
            betas = [params[1], params[3], params[5], params[7]]
            
            # The results (bit strings) of running the circuit 100 times and getting 100 measurements
            bit_strings = []
            for i in range(0, num_reps):
                hold = int(circuit(gammas, betas, edge=None, num_layers=num_layers))
                bit_strings.append(hold) # This appends the integer from 0-15 (if 4 nodes) so it outputs the computational basis measurement in decimal. 

            counts = np.bincount(np.array(bit_strings)) # A 1x16 array that shows the frequency of each bitstring output
            most_freq_bit_string = np.argmax(counts) # Finds the most frequent bitstring

            return counts, bit_strings, most_freq_bit_string

        def get_binary_bit_strings(bit_strings):
            bit_strings_binary = []
            for bit in bit_strings:  # Loops through each of the 100 measurements
                bit_strings_binary.append(decimal_to_binary(bit))
            return bit_strings_binary

        # Cost function
        def cost_function(params):
            bit_strings = get_counts(params)[1]
            binary_bit_strings = get_binary_bit_strings(bit_strings)
            total_cost = 0
            for i in range(0, len(binary_bit_strings)): # Length of binary_bit_strings should be 100
                for edge in edges:
                    start_node = edge[0]
                    end_node = edge[1]
                    weight = edge[2]
                    weighted_cost = -1 * (weight * binary_bit_strings[i][start_node] * (1 - binary_bit_strings[i][end_node]) + weight * binary_bit_strings[i][end_node] * (1 - binary_bit_strings[i][start_node])) 
                    total_cost += weighted_cost
            
            total_cost = float(total_cost) / 100
            print("Cost: "+str(total_cost))
            yield ""
            return total_cost

        nodes = []
        for edge in edges:
            start_node = edge[0]
            end_node = edge[1]
            if (start_node not in nodes):
                nodes.append(start_node)
            if (end_node not in nodes):
                nodes.append(end_node)
        num_nodes = len(nodes)

        params = init_params

        out = minimize(cost_function, x0=params, method="COBYLA", options={'maxiter':num_iters}) 
        # This optimizer changes our initialized params from a 2x4 array into a 1x8 array

        optimal_params = out['x'] # This outputs a 2x4 array not a 1x8 
        optimal_params_vector = []
        for layer in range(len(optimal_params[0])): # Convert the 1x8 array into a 2x4 array
            optimal_params_vector.append(optimal_params[0][layer])
            optimal_params_vector.append(optimal_params[1][layer]) # optimal_params_vector is good
            
        # optimal_params_vector is an array not a tensor 
        final_bitstring = get_counts(optimal_params_vector)

        # The most frequent bitstring is stored in final_bitstring[2]
        binary_bit_string = ''
        for bit in decimal_to_binary(final_bitstring[2]): # This for loop gets the string version of the array binary bit string.
                binary_bit_string += str(bit)
        
        yield f'The answer to our weighted maxcut is {binary_bit_string}'
        # yield '{"maxcut": res}'
    # return jsonify({"maxcut": res})
    return Response(stream_with_context(generate()), mimetype='text/xml')
    # return jsonify({"maxcut": d})

@app.route('/')
def index():
    return jsonify({"Message": "Calculate QAOA"})

if __name__ == "__main__":
    app.run(debug=True)
