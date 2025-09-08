import streamlit as st
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.visualization import plot_bloch_multivector
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Quantum Circuit Simulator")

# --- Gate Definitions ---
# We add display symbols for the grid UI
# CNOT is now represented by Control '●' and Target '⊕'
GATE_DEFINITIONS = {
    'I': {'name': 'Identity', 'matrix': np.array([[1, 0], [0, 1]]), 'params': 0},
    'X': {'name': 'Pauli-X', 'matrix': np.array([[0, 1], [1, 0]]), 'params': 0},
    'Y': {'name': 'Pauli-Y', 'matrix': np.array([[0, -1j], [1j, 0]]), 'params': 0},
    'Z': {'name': 'Pauli-Z', 'matrix': np.array([[1, 0], [0, -1]]), 'params': 0},
    'H': {'name': 'Hadamard', 'matrix': np.array([[1, 1], [1, -1]]) / np.sqrt(2), 'params': 0},
    'S': {'name': 'S Gate', 'matrix': np.array([[1, 0], [0, 1j]]), 'params': 0},
    'T': {'name': 'T Gate', 'matrix': np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]]), 'params': 0},
    '●': {'name': 'Control', 'matrix': None, 'params': 0},
    '⊕': {'name': 'Target (X)', 'matrix': None, 'params': 0},
}
GATE_OPTIONS = list(GATE_DEFINITIONS.keys())

# --- Helper Functions ---
def initialize_state(num_qubits, num_steps):
    """Initializes or resets the circuit grid in the session state."""
    st.session_state.circuit_grid = [['I'] * num_steps for _ in range(num_qubits)]

# --- Streamlit UI ---
st.title('⚛️ Quantum Circuit Simulator')
st.markdown("Visually build a quantum circuit and see the resulting quantum states on Bloch spheres.")

# --- Sidebar Controls ---
with st.sidebar:
    st.header('Circuit Controls')
    
    # Allow user to change number of qubits
    num_qubits = st.slider(
        'Number of Qubits', 1, 5, 2, key='num_qubits_slider'
    )
    
    # Allow user to change number of time steps (circuit depth)
    num_steps = st.slider(
        'Circuit Depth (Steps)', 5, 15, 10, key='num_steps_slider'
    )

    if 'circuit_grid' not in st.session_state or len(st.session_state.circuit_grid) != num_qubits or len(st.session_state.circuit_grid[0]) != num_steps:
        initialize_state(num_qubits, num_steps)

    if st.button('Reset Circuit', use_container_width=True):
        initialize_state(num_qubits, num_steps)
        st.success("Circuit reset!")

# --- Main Circuit Grid UI ---
st.header('Quantum Circuit')
grid_cols = st.columns(num_steps + 1) # +1 for qubit labels

# Header row for time steps
for i in range(num_steps):
    grid_cols[i + 1].markdown(f"<p style='text-align: center;'>{i}</p>", unsafe_allow_html=True)

# Create a row for each qubit
for q in range(num_qubits):
    # Label for the qubit row
    grid_cols[0].markdown(f"`|q{q}⟩`")
    
    # Create a selectbox for each time step in the qubit's row
    for t in range(num_steps):
        # Use a unique key for each selectbox
        gate = grid_cols[t + 1].selectbox(
            f"Q{q}T{t}", 
            options=GATE_OPTIONS, 
            key=f"gate_{q}_{t}",
            label_visibility="collapsed"
        )
        # Update the session state grid when a selection is made
        st.session_state.circuit_grid[q][t] = gate

# --- Execution Logic ---
if st.button('▶️ Execute', type="primary", use_container_width=True):
    try:
        with st.spinner("Simulating circuit..."):
            qc = QuantumCircuit(num_qubits)
            
            # Build the circuit by reading the grid column by column (time step by time step)
            for t in range(num_steps):
                control_qubit = -1
                target_qubit = -1
                
                # First pass for CNOT detection in this time step
                for q in range(num_qubits):
                    gate = st.session_state.circuit_grid[q][t]
                    if gate == '●':
                        if control_qubit != -1: # More than one control
                            raise ValueError(f"Multiple control gates found in time step {t}.")
                        control_qubit = q
                    elif gate == '⊕':
                        if target_qubit != -1: # More than one target
                            raise ValueError(f"Multiple target gates found in time step {t}.")
                        target_qubit = q
                
                # Apply gates for this time step
                if control_qubit != -1 or target_qubit != -1:
                     # This is a CNOT gate step
                    if control_qubit == -1 or target_qubit == -1:
                        raise ValueError(f"Incomplete CNOT gate in time step {t}. Both '●' and '⊕' are required.")
                    qc.cx(control_qubit, target_qubit)
                else:
                    # Apply single-qubit gates for this step
                    for q in range(num_qubits):
                        gate = st.session_state.circuit_grid[q][t]
                        if gate != 'I' and gate != '●' and gate != '⊕':
                             # getattr allows calling a method by its string name, e.g., qc.h(q)
                            getattr(qc, gate.lower())(q)
            
            # Add a barrier to visually separate operations
            qc.barrier()
            
            # --- Simulation ---
            backend = Aer.get_backend('statevector_simulator')
            job = backend.run(qc)
            result = job.result()
            statevector = result.get_statevector()

            st.success("✅ Simulation complete!")

            # --- Display Results ---
            st.header("Quantum States")
            st.markdown("The final state of each qubit is visualized on a Bloch sphere.")
            
            fig = plot_bloch_multivector(statevector)
            st.pyplot(fig)
            plt.close(fig) # Important to close the figure to free up memory

            # Optional: Display the raw statevector
            with st.expander("Show Raw Statevector"):
                st.code(f"{statevector}", language=None)

    except ValueError as e:
        st.error(f"Circuit Error: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
