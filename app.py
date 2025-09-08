import streamlit as st
import numpy as np
# Corrected import statements
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit_aer import Aer
from qiskit.visualization import plot_bloch_multivector
import matplotlib.pyplot as plt

# --- Quantum Gate Definitions (Matrices) ---
# We use NumPy arrays for matrix representation
GATE_MATRICES = {
    'X': {'name': 'Pauli-X', 'symbol': 'X', 'matrix': np.array([[0, 1], [1, 0]])},
    'Y': {'name': 'Pauli-Y', 'symbol': 'Y', 'matrix': np.array([[0, -1j], [1j, 0]])},
    'Z': {'name': 'Pauli-Z', 'symbol': 'Z', 'matrix': np.array([[1, 0], [0, -1]])},
    'H': {'name': 'Hadamard', 'symbol': 'H', 'matrix': np.array([[1, 1], [1, -1]]) / np.sqrt(2)},
    'S': {'name': 'S Gate', 'symbol': 'S', 'matrix': np.array([[1, 0], [0, 1j]])},
    'T': {'name': 'T Gate', 'symbol': 'T', 'matrix': np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])},
    'I': {'name': 'Identity', 'symbol': 'I', 'matrix': np.array([[1, 0], [0, 1]])},
    'CNOT': {'name': 'CNOT', 'symbol': 'CNOT', 'matrix': None, 'is_multi_qubit': True}
}

# --- Streamlit App UI ---
st.set_page_config(layout="wide")
st.title('⚛️ Quantum Circuit Simulator with Streamlit')
st.markdown("""
A simple, interactive quantum circuit simulator built with Streamlit and Qiskit.
Build a circuit by dragging and dropping gates, then click 'Execute' to see the results on the Bloch sphere.
""")

# Initialize session state for the circuit and simulation results
if 'num_qubits' not in st.session_state:
    st.session_state.num_qubits = 2
if 'circuit_data' not in st.session_state:
    st.session_state.circuit_data = []

# --- Sidebar for Controls ---
st.sidebar.header('Circuit Controls')
st.session_state.num_qubits = st.sidebar.slider(
    'Number of Qubits', 1, 8, st.session_state.num_qubits
)

if st.sidebar.button('Reset Circuit'):
    st.session_state.circuit_data = []
    st.experimental_rerun()

# --- Gate Palette ---
st.sidebar.header('Quantum Gates')
gate_palette = st.sidebar.columns(2)
for i, (gate_name, gate_info) in enumerate(GATE_MATRICES.items()):
    col = gate_palette[i % 2]
    with col:
        if st.button(f"{gate_info['symbol']} - {gate_info['name']}"):
            st.session_state.selected_gate = gate_name

# --- Circuit Grid Visualization (simplified) ---
st.header('Circuit Grid')
circuit_columns = st.columns(10) # Create 10 columns for time steps

# This part is a simplified UI. A more complex drag-and-drop would be custom JS
# For now, we'll use buttons to represent the grid
for t in range(10):
    circuit_columns[t].markdown(f"**{t}**")

# This is a placeholder for the circuit visualization
# A full visual representation would require more complex rendering
st.write("Current Circuit:")
st.json(st.session_state.circuit_data)

# --- Execution Logic ---
if st.button('Execute Circuit'):
    if not st.session_state.circuit_data:
        st.warning("Please add some gates to the circuit first!")
    else:
        with st.spinner("Executing simulation..."):
            qc = QuantumCircuit(st.session_state.num_qubits)

            # Apply gates from session state
            for gate in st.session_state.circuit_data:
                gate_type = gate['type']
                qubit = gate['qubit']
                if gate_type == 'CNOT':
                    target = gate['target']
                    qc.cx(qubit, target)
                else:
                    getattr(qc, gate_type.lower())(qubit)

            # Run the simulation
            backend = Aer.get_backend('statevector_simulator')
            job = backend.run(qc)
            result = job.result()
            statevector = result.get_statevector()
            
            st.success("Simulation complete!")

            # Display the results
            st.header('Simulation Results')
            fig = plot_bloch_multivector(statevector)
            st.pyplot(fig)
            st.json(statevector.data)

# --- This is a placeholder for drag-and-drop functionality ---
# You can use Streamlit's components for a more advanced UI, but this shows the basic idea
st.sidebar.info("To add a gate, click the button in the palette.")

# --- Helper functions for the UI logic ---
def place_gate(qubit, time, gate_type):
    # This function is not used in this simplified app, but would be part of a custom UI
    pass


