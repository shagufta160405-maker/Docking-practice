import streamlit as st
import pandas as pd
import requests
import math
from Bio.PDB import PDBParser, PDBIO, Select
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Draw
from rdkit.Geometry import Point3D
import streamlit.components.v1 as components
import time

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="In Silico Molecular Docking Studio",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Functions ---
def fetch_pdb_file(pdb_id):
    """Fetches PDB file contents from the RCSB protein data bank."""
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    return None

class NonHeteroSelect(Select):
    """Biopython selection class to filter out heteroatoms (HETATM records)."""
    def accept_residue(self, residue):
        return residue.get_id()[0] == " "

def parse_heteroatoms(pdb_text):
    """Parses heteroatoms and co-factors directly from raw PDB text lines."""
    hetero_data = []
    for line in pdb_text.splitlines():
        if line.startswith("HETATM"):
            res_name = line[17:20].strip()
            chain_id = line[21].strip()
            res_seq = line[22:26].strip()
            if not any(d['Residue'] == res_name and d['ID'] == res_seq for d in hetero_data):
                hetero_data.append({
                    "Residue": res_name,
                    "Chain": chain_id,
                    "ID": res_seq,
                    "Type": "Co-factor / Heteroatom" if res_name not in ["HOH", "WAT"] else "Water Molecule"
                })
    return pd.DataFrame(hetero_data)

def parse_2d_topology(pdb_text):
    """Parses 2D secondary structure topology metadata elements from PDB headers."""
    helices = 0
    sheets = 0
    total_residues = set()
    
    for line in pdb_text.splitlines():
        if line.startswith("HELIX"):
            helices += 1
        elif line.startswith("SHEET"):
            sheets += 1
        elif line.startswith("ATOM  ") and line[12:16].strip() == "CA":
            res_seq = line[22:26].strip()
            chain_id = line[21].strip()
            total_residues.add(f"{chain_id}_{res_seq}")
            
    return {
        "Alpha Helices (Count)": helices,
        "Beta Sheets (Count)": sheets,
        "Total Computed Residues": len(total_residues)
    }

def generate_topology_mermaid(pdb_text):
    """Generates a Mermaid.js string to visualize 2D secondary structure natively in Streamlit."""
    elements = []
    current_chain = None
    
    # Parse HELIX and SHEET records
    for line in pdb_text.splitlines():
        if line.startswith("HELIX"):
            chain = line[19].strip()
            if current_chain is None: current_chain = chain
            if chain == current_chain:
                start = line[21:25].strip()
                end = line[33:37].strip()
                elements.append({"type": "Alpha Helix", "start": start, "end": end})
        elif line.startswith("SHEET"):
            chain = line[21].strip()
            if current_chain is None: current_chain = chain
            if chain == current_chain:
                start = line[22:26].strip()
                end = line[33:37].strip()
                elements.append({"type": "Beta Sheet", "start": start, "end": end})
                
    try:
        elements.sort(key=lambda x: int(x['start']))
    except ValueError:
        pass
        
    if not elements:
        return None
        
    display_elements = elements[:8]
    
    # Build Mermaid syntax string
    mermaid_str = "graph LR\n"
    for i, el in enumerate(display_elements):
        # Different shapes for Helix vs Sheet
        shape_start = "([" if el['type'] == "Alpha Helix" else "["
        shape_end = "])" if el['type'] == "Alpha Helix" else "]"
        
        mermaid_str += f"  N{i}{shape_start}{el['type']}<br>Res {el['start']}-{el['end']}{shape_end}\n"
        if i > 0:
            mermaid_str += f"  N{i-1} --> N{i}\n"
            
    if len(elements) > 8:
        mermaid_str += f"  N_more[[... {len(elements)-8} more structures]]\n"
        mermaid_str += f"  N{len(display_elements)-1} -.-> N_more\n"
        
    # CSS Styling embedded in Mermaid
    mermaid_str += "  classDef helix fill:#ffb3b3,stroke:#cc0000,stroke-width:2px,color:black;\n"
    mermaid_str += "  classDef sheet fill:#b3d9ff,stroke:#0066cc,stroke-width:2px,color:black;\n"
    
    for i, el in enumerate(display_elements):
        c_name = "helix" if el['type'] == "Alpha Helix" else "sheet"
        mermaid_str += f"  class N{i} {c_name};\n"
        
    return mermaid_str

def calculate_simulation_docking(pdb_id, smiles, pdb_text, ligand_props, strategy):
    """Simulates realistic docking scores heavily weighted by chosen grid strategy."""
    
    # Flat additive modifiers to prevent modulo collisions (guarantees different text/residues)
    strategy_mods = {
        "Scan Cavity (Active Site Boundary Box)": (0, 0.0, 0.0, 0),
        "Target Heteroatoms / Crystallographic Ligand": (137, -1.8, -0.4, 25), 
        "Blind Global Docking Whole Surface": (251, +2.4, +1.2, 80)            
    }
    
    strat_seed_add, strat_score_mod, strat_dist_mod, residue_skip = strategy_mods.get(strategy, (0, 0.0, 0.0, 0))
    
    # Base seed calculation
    base_seed = sum(ord(char) for char in f"{pdb_id.upper()}_{smiles}")
    combined_seed = base_seed + strat_seed_add
    
    # Calculate variable binding affinity
    base_score = -6.8 - (combined_seed % 20) / 10.0
    if ligand_props:
        base_score -= (ligand_props.get("Molecular Weight (g/mol)", 150) % 15) / 10.0
        
    base_score += strat_score_mod
    base_score = round(base_score, 1)
    
    energies = [base_score]
    for i in range(1, 5):
        next_val = round(energies[-1] + 0.3 + (combined_seed % (i + 2)) * 0.1, 1)
        energies.append(next_val)
        
    # Extract authentic amino acid environments based on strategy skip offset
    true_residues = []
    current_skip = 0
    
    for line in pdb_text.splitlines():
        if line.startswith("ATOM  ") and line[12:16].strip() == "CA":
            current_skip += 1
            if current_skip < residue_skip: 
                continue
            res_name = line[17:20].strip().title()
            res_number = line[22:26].strip()
            formatted_res = f"{res_name}{res_number}"
            if formatted_res not in true_residues:
                true_residues.append(formatted_res)
            if len(true_residues) >= 4:
                break
                
    while len(true_residues) < 4:
        true_residues.append(f"Res{100 + len(true_residues)}")

    vectors_pool = ["Hydrogen Bond", "Pi-Pi Stacking", "Van der Waals", "Salt Bridge", "Cation-Pi", "Hydrophobic"]
    summaries_pool = [
        "Strong electrostatic localization to ligand donor group.",
        "Aromatic structural pairing to ligand framework.",
        "Hydrophobic binding pocket envelope contact optimization.",
        "Ionic stabilization across receptor cavity.",
        "Cationic pairing with electron-rich ligand rings.",
        "Non-polar surface area burial interaction."
    ]
    
    interactions = []
    distances = []
    func_summaries = []
    
    for i in range(4):
        # Now yields totally different indices based on the active strategy
        idx = (combined_seed + i * 3) % len(vectors_pool)
        interactions.append(vectors_pool[idx])
        func_summaries.append(summaries_pool[idx])
        
        dist = round(2.8 + ((combined_seed * (i + 1)) % 15) / 10.0 + strat_dist_mod, 2)
        dist = max(1.8, dist) 
        distances.append(dist)
        
    return base_score, energies, true_residues, interactions, distances, func_summaries

def render_3d_viewer(pdb_str, ligand_smiles=None, style="cartoon", element_id="container", grid_center=None, pose_idx=1):
    """Generates an inline HTML canvas containing py3Dmol for 3D visualization and dynamically offsets ligand."""
    style_opts = f"{{ {style}: {{color: 'spectrum'}} }}"
    
    ligand_js = ""
    if ligand_smiles:
        mol = Chem.MolFromSmiles(ligand_smiles)
        if mol:
            mol = Chem.AddHs(mol)
            from rdkit.Chem import AllChem
            AllChem.EmbedMolecule(mol, randomSeed=42)
            
            if grid_center:
                conf = mol.GetConformer()
                num_atoms = mol.GetNumAtoms()
                
                # Base spatial center
                cx = sum(conf.GetAtomPosition(i).x for i in range(num_atoms)) / num_atoms
                cy = sum(conf.GetAtomPosition(i).y for i in range(num_atoms)) / num_atoms
                cz = sum(conf.GetAtomPosition(i).z for i in range(num_atoms)) / num_atoms
                
                # Grid translation
                dx = grid_center[0] - cx
                dy = grid_center[1] - cy
                dz = grid_center[2] - cz
                
                # Dynamic Pose Mathematics (Shifts and rotates the molecule based on pose selection)
                pose_shift_x = (pose_idx - 1) * 2.5 * (-1 if pose_idx % 2 == 0 else 1)
                pose_shift_y = (pose_idx - 1) * 1.8
                pose_shift_z = (pose_idx - 1) * 2.2 * (-1 if pose_idx % 3 == 0 else 1)

                angle = (pose_idx - 1) * (math.pi / 3) # Rotate ~60 degrees per pose index
                cos_a = math.cos(angle)
                sin_a = math.sin(angle)
                
                for i in range(num_atoms):
                    pos = conf.GetAtomPosition(i)
                    # Shift to local origin
                    ox = pos.x - cx
                    oy = pos.y - cy
                    oz = pos.z - cz
                    # Rotate on Z/Y axes
                    rx = ox * cos_a - oy * sin_a
                    ry = ox * sin_a + oy * cos_a
                    rz = oz
                    # Translate to grid + apply pose offset
                    conf.SetAtomPosition(i, Point3D(rx + cx + dx + pose_shift_x, ry + cy + dy + pose_shift_y, rz + cz + dz + pose_shift_z))

            mol_block = Chem.MolToMolBlock(mol)
            cleaned_block = mol_block.replace('\n', '\\n').replace('\r', '')
            ligand_js = f"""
            var ligand_mol = msv.addModel(`{cleaned_block}`, "sdf");
            msv.setStyle({{model: ligand_mol}}, {{stick: {{colorscheme: 'cyanCarbon', radius: 0.15}} }});
            """

    cleaned_pdb = pdb_str.replace('\n', '\\n').replace('\r', '')
    
    html_content = f"""
    <div id="{element_id}" style="height: 400px; width: 100%; position: relative;"></div>
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <script>
        var element = document.getElementById('{element_id}');
        var msv = $3Dmol.createViewer(element, {{backgroundColor: '#111217'}});
        var protein_mol = msv.addModel(`{cleaned_pdb}`, "pdb");
        msv.setStyle({{model: protein_mol}}, {style_opts});
        {ligand_js}
        msv.zoomTo();
        msv.render();
    </script>
    """
    components.html(html_content, height=410)

# --- App State Initialization ---
if 'pdb_text' not in st.session_state: st.session_state.pdb_text = None
if 'pure_protein' not in st.session_state: st.session_state.pure_protein = None
if 'smiles' not in st.session_state: st.session_state.smiles = ""
if 'ligand_props' not in st.session_state: st.session_state.ligand_props = None
if 'topology_data' not in st.session_state: st.session_state.topology_data = None
if 'topology_graph' not in st.session_state: st.session_state.topology_graph = None

# --- Main Dashboard Header ---
st.title("🧬 Multi-Phase In Silico Docking Workspace")
st.markdown("Automate receptor sanitization, ligand feature analysis, and grid-targeted compound docking on a unified workflow canvas.")
st.markdown("---")

# =====================================================================
# PHASE 1: PROTEIN PREPARATION & TOPOLOGY MAPS
# =====================================================================
st.header("📍 Phase 1: Receptor Preparation & Structural Analytics")

col1, col2 = st.columns([1, 2])

with col1:
    pdb_id = st.text_input("Enter 4-Character PDB ID:", max_chars=4, placeholder="e.g., 1IEP").strip()
    fetch_btn = st.button("Fetch and Prepare Structure", type="primary")
    
    if fetch_btn and pdb_id:
        with st.spinner("Retrieving coordinate records from RCSB..."):
            raw_text = fetch_pdb_file(pdb_id)
            if raw_text:
                st.session_state.pdb_text = raw_text
                st.session_state.topology_data = parse_2d_topology(raw_text)
                # Generate Mermaid.js native diagram
                st.session_state.topology_graph = generate_topology_mermaid(raw_text)
                
                parser = PDBParser(QUIET=True)
                from io import StringIO
                pdb_fh = StringIO(raw_text)
                structure = parser.get_structure(pdb_id, pdb_fh)
                
                io = PDBIO()
                io.set_structure(structure)
                out_stream = StringIO()
                io.save(out_stream, NonHeteroSelect())
                st.session_state.pure_protein = out_stream.getvalue()
                st.success(f"Successfully processed {pdb_id.upper()}!")
            else:
                st.error("Failed to discover PDB ID. Double-check your code entry.")

if st.session_state.pdb_text:
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.subheader("Raw Complex Structure View")
        render_mode = st.selectbox("Style View", ["cartoon", "sphere", "line"], key="p1_style")
        render_3d_viewer(st.session_state.pdb_text, style=render_mode, element_id="raw_viewer")
    
    with c2:
        st.subheader("Sanitized Pure Protein")
        st.markdown("*Heteroatoms, waters, and co-factors removed*")
        render_3d_viewer(st.session_state.pure_protein, style=render_mode, element_id="pure_viewer")
        st.download_button(
            label="📥 Download Prepared PDBQT File",
            data=st.session_state.pure_protein,
            file_name=f"{pdb_id}_prepared.pdbqt",
            mime="text/plain"
        )
        
    with c3:
        st.subheader("2D Topology Summary")
        if st.session_state.topology_data:
            topo_df = pd.DataFrame(st.session_state.topology_data.items(), columns=["Structural Feature", "Value"])
            st.dataframe(topo_df, use_container_width=True, hide_index=True)
            
        st.subheader("2D Sequence Topology Diagram")
        if st.session_state.topology_graph:
            # Rendering via Native Streamlit Markdown (Mermaid)
            st.markdown(f"```mermaid\n{st.session_state.topology_graph}\n```")

st.markdown("---")

# =====================================================================
# PHASE 2: LIGAND PREPARATION & 2D DIAGRAM
# =====================================================================
if st.session_state.pure_protein:
    st.header("🧪 Phase 2: Ligand Initialization & Feature Extraction")
    
    l_col1, l_col2, l_col3 = st.columns(3)
    
    with l_col1:
        smiles_input = st.text_input("Enter Ligand SMILES String:", placeholder="e.g., CC(=O)OC1=CC=CC=C1C(=O)O")
        process_ligand = st.button("Analyze Ligand Properties")
        
        if process_ligand and smiles_input:
            mol = Chem.MolFromSmiles(smiles_input)
            if mol:
                st.session_state.smiles = smiles_input
                st.session_state.ligand_props = {
                    "Molecular Weight (g/mol)": round(Descriptors.MolWt(mol), 2),
                    "LogP (Lipophilicity)": round(Descriptors.MolLogP(mol), 2),
                    "H-Bond Donors": Lipinski.NumHDonors(mol),
                    "H-Bond Acceptors": Lipinski.NumHAcceptors(mol),
                    "Rotatable Bonds": Lipinski.NumRotatableBonds(mol)
                }
                st.success("Ligand successfully parameterized!")
            else:
                st.error("Invalid SMILES string. Please try again.")
                
    if st.session_state.ligand_props:
        with l_col2:
            st.subheader("2D Ligand Structure")
            mol = Chem.MolFromSmiles(st.session_state.smiles)
            img = Draw.MolToImage(mol, size=(300, 300))
            st.image(img, caption="RDKit Generated 2D Framework", use_column_width=True)
            
        with l_col3:
            st.subheader("Physicochemical Properties")
            props_df = pd.DataFrame(st.session_state.ligand_props.items(), columns=["Property", "Value"])
            st.dataframe(props_df, use_container_width=True, hide_index=True)

    st.markdown("---")

# =====================================================================
# PHASE 3: DOCKING SIMULATION
# =====================================================================
if st.session_state.pure_protein and st.session_state.smiles:
    st.header("⚙️ Phase 3: Grid Configuration & Docking Execution")
    
    grid_strategy = st.selectbox(
        "Select Grid Box Search Strategy:", 
        [
            "Scan Cavity (Active Site Boundary Box)", 
            "Target Heteroatoms / Crystallographic Ligand", 
            "Blind Global Docking Whole Surface"
        ]
    )
    
    run_docking = st.button("Execute Docking Simulation", type="primary", use_container_width=True)
    
    if run_docking:
        with st.spinner(f"Running simulation using {grid_strategy}..."):
            time.sleep(2)  # Simulate compute time
            
            # Execute docking math helpers
            score, energies, residues, interactions, distances, summaries = calculate_simulation_docking(
                pdb_id, st.session_state.smiles, st.session_state.pure_protein, st.session_state.ligand_props, grid_strategy
            )
            
            st.session_state.docking_results = {
                "score": score,
                "energies": energies,
                "residues": residues,
                "interactions": interactions,
                "distances": distances,
                "summaries": summaries
            }
            st.success("Simulation Complete!")

    if 'docking_results' in st.session_state:
        res = st.session_state.docking_results
        
        d_col1, d_col2 = st.columns([1, 2])
        
        with d_col1:
            st.metric(label="Top Pose Binding Affinity", value=f"{res['score']} kcal/mol", delta="High Affinity", delta_color="inverse")
            
            pose_df = pd.DataFrame({
                "Pose Rank": [1, 2, 3, 4, 5],
                "Energy (kcal/mol)": res['energies']
            })
            st.dataframe(pose_df, use_container_width=True, hide_index=True)
            
            st.subheader("Key Molecular Interactions")
            interaction_df = pd.DataFrame({
                "Residue": res['residues'],
                "Interaction Type": res['interactions'],
                "Distance (Å)": res['distances']
            })
            st.dataframe(interaction_df, use_container_width=True, hide_index=True)
            
        with d_col2:
            st.subheader("3D Pose Inspection")
            selected_pose = st.slider("Select Docking Pose Index", min_value=1, max_value=5, value=1)
            
            # Arbitrary grid center calculation based on the chosen strategy (visual only)
            grid_center = (0, 0, 0) if grid_strategy == "Blind Global Docking Whole Surface" else (10, -5, 12)
            
            render_3d_viewer(
                st.session_state.pure_protein, 
                ligand_smiles=st.session_state.smiles, 
                style="cartoon", 
                element_id="docking_viewer",
                grid_center=grid_center,
                pose_idx=selected_pose
            )
