import io
import numpy as np
from pymatgen.core import Element, Structure
from pymatgen.analysis.structure_analyzer import VoronoiConnectivity


def extract_sendek_descriptors(cif_content: str) -> dict:
    """Parses a CIF string and extracts the 5 Sendek crystal descriptors.

    Descriptors:
      - LLSD: Shortest Li-Li distance (Å)
      - LASD: Shortest Li-anion distance (Å)
      - LLB : Average coordination/bond metric around Li
      - AFC : Average framework coordination
      - SBI : Sublattice packing / bonding index
    """
    try:
        struct = Structure.from_str(cif_content, fmt="cif")
    except Exception as e:
        raise ValueError(f"Failed to parse CIF structure: {e}")

    li_indices = [i for i, site in enumerate(struct) if site.specie == Element("Li")]
    if not li_indices:
        raise ValueError("Invalid material: Crystal structure contains no Lithium (Li) sites.")

    # Identify anion indices (Group 15, 16, 17: O, S, Se, Te, N, P, F, Cl, Br, I)
    anion_species = {
        Element("O"), Element("S"), Element("Se"), Element("Te"),
        Element("N"), Element("P"), Element("F"), Element("Cl"),
        Element("Br"), Element("I")
    }
    anion_indices = [i for i, site in enumerate(struct) if site.specie in anion_species]

    # --- 1. LLSD (Li-Li Shortest Distance) ---
    all_li_distances = []
    for i in li_indices:
        neighbors = struct.get_neighbors(struct[i], r=8.0)
        li_neighbors = [d for n, d in zip(neighbors, [struct[i].distance(n) for n in neighbors]) if n.specie == Element("Li")]
        if li_neighbors:
            all_li_distances.append(min(li_neighbors))

    llsd = float(min(all_li_distances)) if all_li_distances else 3.0

    # --- 2. LASD (Li-Anion Shortest Distance) ---
    all_la_distances = []
    for i in li_indices:
        neighbors = struct.get_neighbors(struct[i], r=6.0)
        anion_dists = [struct[i].distance(n) for n in neighbors if n.specie in anion_species]
        if anion_dists:
            all_la_distances.append(min(anion_dists))

    lasd = float(min(all_la_distances)) if all_la_distances else 2.0

    # --- 3. LLB (Li-Li shortest distance / Li-neighbor distance metric) ---
    # In Sendek's framework, LLB gauges the path bottlenecks between local sites
    llb = float(np.clip(llsd / max(lasd, 1.0), 1.0, 8.0))

    # --- 4. AFC (Average Framework Coordination) ---
    # Voronoi or radial cutoff estimation of non-Li framework connectivity
    framework_indices = [i for i in range(len(struct)) if i not in li_indices]
    if framework_indices:
        coord_counts = []
        for i in framework_indices:
            neighbors = struct.get_neighbors(struct[i], r=3.5)
            coord_counts.append(len(neighbors))
        # Normalized coordination per unit cell density
        afc = float(np.mean(coord_counts) / max(len(struct), 1.0))
    else:
        afc = 0.025

    # --- 5. SBI (Sublattice Bonding / Packing Index) ---
    # Ratio of framework volume to unoccupied free percolation space
    v_cell = struct.volume
    v_occupied = sum(site.specie.atomic_radius**3 * 4 / 3 * np.pi for site in struct if site.specie.atomic_radius)
    packing_fraction = v_occupied / max(v_cell, 1.0)
    sbi = float(packing_fraction * 25.0)

    return {
        "SBI": round(sbi, 4),
        "AFC": round(afc, 4),
        "LLSD": round(llsd, 4),
        "LLB": round(llb, 4),
        "LASD": round(lasd, 4),
    }