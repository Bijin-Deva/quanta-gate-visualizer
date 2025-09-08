import streamlit as st
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import Aer
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from qiskit.quantum_info import DensityMatrix, partial_trace

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Quantum Circuit Simulator")

# --- Gate Definitions ---
# CNOT is represented by Control '●' and Target '⊕'
GATE_DEFINITIONS = {
    'I': {'name': 'Identity'},
    'X': {'name': 'Pauli-X'},
    'Y': {'name': 'Pauli-Y'},
    'Z': {'name': 'Pauli-Z'},
    'H': {'name': 'Hadamard'},
    'S': {'name': 'S Gate'},
    'T': {'name': 'T Gate'},
    '●': {'name': 'Control'},
    '⊕': {'name': 'Target (X)'},
}
GATE_OPTIONS = list(GATE_DEFINITIONS.keys())

# --- Helper Functions ---
def initialize_state(num_qubits, num_steps):
    """Initializes or resets the circuit grid in the session state."""
    st.session_state.circuit_grid = [['I'] * num_steps for _ in range(num_qubits)]

def create_interactive_bloch_sphere(bloch_vector, title=""):
    """Creates an interactive Bloch sphere plot using Plotly."""
    x, y, z = bloch_vector

    fig = go.Figure()

    # Draw the sphere surface
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    sphere_x = np.outer(np.cos(u), np.sin(v))
    sphere_y = np.outer(np.sin(u), np.sin(v))
    sphere_z = np.outer(np.ones(np.size(u)), np.cos(v))
    fig.add_trace(go.Surface(x=sphere_x, y=sphere_y, z=sphere_z,
                             colorscale=[[0, 'lightblue'], [1, 'lightblue']],
                             opacity=0.3,
                             showscale=False,
                             name="Sphere"))

    # Draw the axes
    fig.add_trace(go.Scatter3d(x=[-1.2, 1.2], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='grey'), name='X-axis'))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[-1.2, 1.2], z=[0, 0], mode='lines', line=dict(color='grey'), name='Y-axis'))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[-1.2, 1.2], mode='lines', line=dict(color='grey'), name='Z-axis'))

    # Draw the state vector with a cone arrowhead
    fig.add_trace(go.Cone(x=[x], y=[y], z=[z], u=[x], v=[y], w=[z],
                          sizemode="absolute", sizeref=0.1,
                          anchor="tip",
                          showscale=False,
                          colorscale=[[0, 'red'], [1, 'red']],
                          name="State Vector"))

    # Update layout
    fig.update_layout(
        title=dict(text=title, x=0.5),
        showlegend=False,
        scene=dict(
            xaxis=dict(title='X', range=[-1.2, 1.2], showticklabels=False, showgrid=False, zeroline=False, backgroundcolor="rgba(0,0,0,0)"),
            yaxis=dict(title='Y', range=[-1.2, 1.2], showticklabels=False, showgrid=False, zeroline=False, backgroundcolor="rgba(0,0,0,0)"),
            zaxis=dict(title='Z', range=[-1.2, 1.2], showticklabels=False, showgrid=False, zeroline=False, backgroundcolor="rgba(0,0,0,0)"),
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig

# --- Streamlit UI ---
st.title('⚛️ Quantum Circuit Simulator')
st.markdown("Visually build a quantum circuit and see the resulting quantum states on interactive Bloch spheres.")

# --- Sidebar Controls ---
with st.sidebar:
    st.header('Circuit Controls')
    
    num_qubits = st.slider('Number of Qubits', 1, 5, 2, key='num_qubits_slider')
    num_steps = st.slider('Circuit Depth (Steps)', 5, 15, 10, key='num_steps_slider')

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
    grid_cols[0].markdown(f"`|q{q}⟩`")
    for t in range(num_steps):
        gate = grid_cols[t + 1].selectbox(
            f"Q{q}T{t}", 
            options=GATE_OPTIONS, 
            key=f"gate_{q}_{t}",
            label_visibility="collapsed"
        )
        st.session_state.circuit_grid[q][t] = gate

# --- Execution Logic ---
if st.button('▶️ Execute', type="primary", use_container_width=True):
    try:
        with st.spinner("Simulating circuit..."):
            qc = QuantumCircuit(num_qubits)
            for t in range(num_steps):
                control_qubit = -1
                target_qubit = -1
                for q in range(num_qubits):
                    gate = st.session_state.circuit_grid[q][t]
                    if gate == '●':
                        if control_qubit != -1: raise ValueError(f"Multiple control gates found in time step {t}.")
                        control_qubit = q
                    elif gate == '⊕':
                        if target_qubit != -1: raise ValueError(f"Multiple target gates found in time step {t}.")
                        target_qubit = q
                
                if control_qubit != -1 or target_qubit != -1:
                    if control_qubit == -1 or target_qubit == -1: raise ValueError(f"Incomplete CNOT gate in time step {t}.")
                    qc.cx(control_qubit, target_qubit)
                else:
                    for q in range(num_qubits):
                        gate = st.session_state.circuit_grid[q][t]
                        if gate != 'I' and gate != '●' and gate != '⊕':
                            getattr(qc, gate.lower())(q)
            qc.barrier()

            # --- Simulation ---
            backend = Aer.get_backend('statevector_simulator')
            job = backend.run(qc)
            result = job.result()
            statevector = result.get_statevector()
            
            st.success("✅ Simulation complete!")

            # --- Display Results ---
            st.header("Quantum States")
            st.markdown("The final state of each qubit is visualized on an interactive Bloch sphere. You can click and drag to rotate them.")

            # Calculate Bloch vectors for each qubit
            dm = DensityMatrix(statevector)
            bloch_vectors = []
            pauli_x = np.array([[0, 1], [1, 0]])
            pauli_y = np.array([[0, -1j], [1j, 0]])
            pauli_z = np.array([[1, 0], [0, -1]])

            for i in range(num_qubits):
                reduced_dm = partial_trace(dm, [q for q in range(num_qubits) if q != i])
                x = np.real(np.trace(reduced_dm.data @ pauli_x))
                y = np.real(np.trace(reduced_dm.data @ pauli_y))
                z = np.real(np.trace(reduced_dm.data @ pauli_z))
                bloch_vectors.append([x, y, z])

            # Display interactive Bloch spheres in columns
            cols = st.columns(num_qubits)
            for i, vec in enumerate(bloch_vectors):
                with cols[i]:
                    fig = create_interactive_bloch_sphere(vec, title=f"Qubit {i}")
                    st.plotly_chart(fig, use_container_width=True)

            # Optional: Display the raw statevector
            with st.expander("Show Raw Statevector"):
                st.code(f"{statevector}", language=None)

    except ValueError as e:
        st.error(f"Circuit Error: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
