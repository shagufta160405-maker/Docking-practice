import streamlit as st
import subprocess
import os
from stmol import showmol
import py3Dmol

st.title("Molecular Docking App")

# 1. Uploads
protein = st.file_uploader("Upload Protein (.pdb)", type="pdb")
ligand = st.file_uploader("Upload Ligand (.sdf)", type="sdf")

if protein and ligand:
    # Save files locally
    with open("receptor.pdb", "wb") as f: f.write(protein.getbuffer())
    with open("ligand.sdf", "wb") as f: f.write(ligand.getbuffer())
    
    # 2. Visualization (Phase 2)
    view = py3Dmol.view()
    view.addModel(open("receptor.pdb").read(), 'pdb')
    view.setStyle({'cartoon': {'color': 'spectrum'}})
    showmol(view, height=400, width=700)

    # 3. Docking (Phase 3 - Invisible Subprocess)
    if st.button("Run Vina Docking"):
        with st.spinner("Docking in progress..."):
            # Command to run Vina without opening a terminal window
            cmd = [
                "./vina", 
                "--receptor", "receptor.pdb", # Ideally convert to .pdbqt first
                "--ligand", "ligand.sdf", 
                "--out", "result.pdbqt",
                "--center_x", "0", "--center_y", "0", "--center_z", "0",
                "--size_x", "20", "--size_y", "20", "--size_z", "20"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            st.success("Docking Complete!")
            st.text(result.stdout) # Shows Vina log in the app
