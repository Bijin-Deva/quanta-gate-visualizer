import streamlit as st
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import Aer
import plotly.graph_objects as go
from qiskit.quantum_info import DensityMatrix, partial_trace

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Quantum Circuit Simulator")

# --- Gate Definitions ---
GATE_DEFINITIONS = {
    'I': {'name': 'Identity', 'color': '#6c757d'},
    'H': {'name': 'Hadamard', 'color': '#0d6efd'},
    'X': {'name': 'Pauli-X', 'color': '#dc3545'},
    'Y': {'name': 'Pauli-Y', 'color': '#dc3545'},
    'Z': {'name': 'Pauli-Z', 'color': '#dc3545'},
    'S': {'name': 'S Gate', 'color': '#ffc107'},
    'T': {'name': 'T Gate', 'color': '#ffc107'},
    '●': {'name': 'Control', 'color': '#198754'},
    '⊕': 'Target (X)', # Special case for display
}

# --- Helper Functions & State Management ---

def initialize_state(num_qubits, num_steps):
    """Initializes or resets the circuit grid and active gate."""
    st.session_state.circuit_grid = [['I'] * num_steps for _ in range(num_qubits)]
    if 'active_gate' not in st.session_state:
        st.session_state.active_gate = 'H'

def set_active_gate(gate_symbol):
    """Callback to set the currently selected gate."""
    st.session_state.active_gate = gate_symbol

def place_gate(q, t):
    """Callback to place the active gate on the grid."""
    active = st.session_state.active_gate
    # Logic to handle placing Control and Target for CNOT
    if active == 'CNOT':
        # If placing CNOT, first place Control, then user must select Target
        st.session_state.circuit_grid[q][t] = '●'
        st.session_state.active_gate = '⊕' # Immediately switch to Target mode
    elif active == '⊕':
         st.session_state.circuit_grid[q][t] = '⊕'
         st.session_state.active_gate = 'H' # Reset to a default gate after placing
    else:
        st.session_state.circuit_grid[q][t] = active

def create_interactive_bloch_sphere(bloch_vector, title=""):
    """Creates an interactive Bloch sphere plot using Plotly."""
    x, y, z = bloch_vector
    fig = go.Figure()
    # Draw the sphere surface
    u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:100j]
    sphere_x = np.cos(u) * np.sin(v)
    sphere_y = np.sin(u) * np.sin(v)
    sphere_z = np.cos(v)
    fig.add_trace(go.Surface(x=sphere_x, y=sphere_y, z=sphere_z,
                             colorscale=[[0, 'lightblue'], [1, 'lightblue']],
                             opacity=0.3, showscale=False))
    # Draw axes and state vector
    fig.add_trace(go.Scatter3d(x=[-1.2, 1.2], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='grey')))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[-1.2, 1.2], z=[0, 0], mode='lines', line=dict(color='grey')))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[-1.2, 1.2], mode='lines', line=dict(color='grey')))
    fig.add_trace(go.Cone(x=[x], y=[y], z=[z], u=[x], v=[y], w=[z],
                          sizemode="absolute", sizeref=0.1, anchor="tip",
                          showscale=False, colorscale=[[0, 'red'], [1, 'red']]))
    fig.update_layout(
        title=dict(text=title, x=0.5), showlegend=False,
        scene=dict(xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, backgroundcolor="rgba(0,0,0,0)"),
                   yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, backgroundcolor="rgba(0,0,0,0)"),
                   zaxis=dict(showticklabels=False, showgrid=False, zeroline=False, backgroundcolor="rgba(0,0,0,0)"),
                   aspectmode='cube'),
        margin=dict(l=0, r=0, b=0, t=40))
    return fig

# --- Streamlit UI ---
st.title('⚛️ Quantum Circuit Simulator')
st.markdown("Select a gate from the sidebar, then click on the grid to place it.")

# --- Sidebar ---
with st.sidebar:
    st.header('Circuit Controls')
    num_qubits = st.slider('Number of Qubits', 1, 5, 2, key='num_qubits_slider')
    num_steps = st.slider('Circuit Depth', 5, 15, 10, key='num_steps_slider')

    # Initialize state if it doesn't exist or if dimensions changed
    if 'circuit_grid' not in st.session_state or len(st.session_state.circuit_grid) != num_qubits or len(st.session_state.circuit_grid[0]) != num_steps:
        initialize_state(num_qubits, num_steps)

    if st.button('Reset Circuit', use_container_width=True):
        initialize_state(num_qubits, num_steps)
        st.success("Circuit reset!")

    st.header("Gate Palette")
    st.write("Current Gate: **" + st.session_state.active_gate + "**")
    
    # Gate selection buttons
    gate_palette_cols = st.columns(2)
    palette_gates = ['H', 'X', 'Y', 'Z', 'S', 'T', 'I', 'CNOT']
    for i, gate in enumerate(palette_gates):
        gate_palette_cols[i % 2].button(
            gate, on_click=set_active_gate, args=(gate,), use_container_width=True
        )
    if st.session_state.active_gate == '⊕':
        st.info("Now, click a grid cell to place the CNOT Target (⊕).")


# --- Main Circuit Grid UI ---
st.header('Quantum Circuit')
grid_cols = st.columns(num_steps + 1)
grid_cols[0].markdown("---") # Spacer

for i in range(num_steps):
    grid_cols[i + 1].markdown(f"<p style='text-align: center;'>{i}</p>", unsafe_allow_html=True)

for q in range(num_qubits):
    grid_cols[0].markdown(f"`|q{q}⟩`")
    for t in range(num_steps):
        gate_in_cell = st.session_state.circuit_grid[q][t]
        grid_cols[t + 1].button(
            gate_in_cell, key=f"cell_{q}_{t}", on_click=place_gate, args=(q, t), use_container_width=True
        )


# --- Execution Logic ---
if st.button('▶️ Execute', type="primary", use_container_width=True):
    try:
        with st.spinner("Simulating circuit..."):
            qc = QuantumCircuit(num_qubits)
            # Build the circuit from the grid state
            for t in range(num_steps):
                control_qubit = -1
                target_qubit = -1
                for q in range(num_qubits):
                    gate = st.session_state.circuit_grid[q][t]
                    if gate == '●':
                        control_qubit = q
                    elif gate == '⊕':
                        target_qubit = q
                
                if control_qubit != -1 and target_qubit != -1:
                    qc.cx(control_qubit, target_qubit)
                elif control_qubit != -1 or target_qubit != -1:
                    raise ValueError(f"Incomplete CNOT gate in time step {t}.")
                else:
                    for q in range(num_qubits):
                        gate = st.session_state.circuit_grid[q][t]
                        if gate != 'I':
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
            dm = DensityMatrix(statevector)
            bloch_vectors = []
            for i in range(num_qubits):
                reduced_dm = partial_trace(dm, [q for q in range(num_qubits) if q != i])
                x = np.real(np.trace(reduced_dm.data @ np.array([[0, 1], [1, 0]])))
                y = np.real(np.trace(reduced_dm.data @ np.array([[0, -1j], [1j, 0]])))
                z = np.real(np.trace(reduced_dm.data @ np.array([[1, 0], [0, -1]])))
                bloch_vectors.append([x, y, z])
            
            cols = st.columns(num_qubits)
            for i, vec in enumerate(bloch_vectors):
                with cols[i]:
                    fig = create_interactive_bloch_sphere(vec, title=f"Qubit {i}")
                    st.plotly_chart(fig, use_container_width=True)

    except ValueError as e:
        st.error(f"Circuit Error: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
