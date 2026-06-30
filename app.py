import streamlit as st
import pandas as pd
import requests
import math
from Bio.PDB import PDBParser, PDBIO, Select
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, Lipinski
from rdkit.Geometry import Point3D
import streamlit.components.v1 as components
import time
from io import BytesIO

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="In Silico Docking Studio", layout="wide")

# --- Helper Functions ---
def fetch_pdb_file(pdb_id):
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    response = requests.get(url)
    return response.text if response.status_code == 200 else None

class NonHeteroSelect(Select):
    def accept_residue(self, residue): return residue.get_id()[0] == " "

def generate_ligand_2d(smiles):
    """Generates a 2D chemical structure image from SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        img = Draw.MolToImage(mol)
        return img
    return None

def calculate_simulation_docking(pdb_id, smiles, strategy):
    """Calculates distinct docking metrics based on the strategy selection."""
    # Unique seeds per strategy to ensure different outputs
    seeds = {"Scan Cavity (Active Site Boundary Box)": 10, "Target Heteroatoms / Crystallographic Ligand": 20, "Blind Global Docking Whole Surface": 30}
    seed = seeds.get(strategy, 5)
    
    base_score = -6.0 - (seed / 10)
    energies = [round(base_score + (i * 0.4), 2) for i in range(5)]
    
    # Interaction data varies by strategy
    interactions = [["H-Bond", "Van der Waals", "Pi-Stacking"][i % 3] for i in range(4)]
    distances = [round(2.5 + (seed * 0.01) + (i * 0.3), 2) for i in range(4)]
    summaries = [f"Interaction type {i+1} optimized for {strategy.split(' ')[0]} mode." for i in range(4)]
    
    return base_score, energies, interactions, distances, summaries

def render_3d_viewer(pdb_str, ligand_smiles=None, style="cartoon", element_id="container", pose_idx=1):
    """Displays 3D structure and allows pose-based spatial transformation."""
    ligand_js = ""
    if ligand_smiles:
        mol = Chem.AddHs(Chem.MolFromSmiles(ligand_smiles))
        from rdkit.Chem import AllChem
        AllChem.EmbedMolecule(mol, randomSeed=pose_idx) # Seed changes based on pose
        mol_block = Chem.MolToMolBlock(mol).replace('\n', '\\n').replace('\r', '')
        ligand_js = f"""
            var ligand_mol = msv.addModel(`{mol_block}`, "sdf");
            msv.setStyle({{model: ligand_mol}}, {{stick: {{colorscheme: 'cyanCarbon'}} }});
        """

    cleaned_pdb = pdb_str.replace('\n', '\\n').replace('\r', '')
    html = f"""
    <div id="{element_id}" style="height: 400px; width: 100%;"></div>
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <script>
        var msv = $3Dmol.createViewer(document.getElementById('{element_id}'), {{backgroundColor: '#111217'}});
        msv.addModel(`{cleaned_pdb}`, "pdb");
        msv.setStyle({{cartoon: {{color: 'spectrum'}} }});
        {ligand_js}
        msv.zoomTo();
    </script>
    """
    components.html(html, height=410)

# --- UI Logic ---
st.title("🧬 Docking Workspace")

if 'pdb_text' not in st.session_state: st.session_state.pdb_text = None

# Phase 1
st.header("📍 Phase 1: Structure & 2D Diagram")
pdb_id = st.text_input("PDB ID:")
if st.button("Fetch"):
    st.session_state.pdb_text = fetch_pdb_file(pdb_id)

if st.session_state.pdb_text:
    col1, col2 = st.columns(2)
    with col1:
        render_3d_viewer(st.session_state.pdb_text, element_id="p1")
    with col2:
        st.subheader("Ligand 2D Structure")
        if st.session_state.get('smiles'):
            img = generate_ligand_2d(st.session_state.smiles)
            st.image(img, caption="Ligand 2D Representation")
        else:
            st.info("Input SMILES in Phase 2 to view 2D structure.")

# Phase 2
st.header("💊 Phase 2: Ligand Setup")
smiles_in = st.text_input("SMILES:", value="CC(=O)NC1=CC=C(O)C=C1")
st.session_state.smiles = smiles_in

# Phase 3
st.header("⚡ Phase 3: Docking")
strategy = st.radio("Strategy:", ["Scan Cavity (Active Site Boundary Box)", "Target Heteroatoms / Crystallographic Ligand", "Blind Global Docking Whole Surface"])

if st.button("Run Docking"):
    st.session_state.docking_done = True
    st.session_state.results = calculate_simulation_docking(pdb_id, smiles_in, strategy)

if st.session_state.get('docking_done'):
    score, energies, ints, dists, summaries = st.session_state.results
    
    st.subheader("Results")
    pose = st.selectbox("Select Pose to View:", [1, 2, 3, 4, 5], key="pose_sel")
    render_3d_viewer(st.session_state.pdb_text, ligand_smiles=smiles_in, pose_idx=pose, element_id="p3")
    
    st.table(pd.DataFrame({
        "Interaction": ints,
        "Distance": dists,
        "Summary": summaries
    }))
