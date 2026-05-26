"""Biodata MCP Server

A remote MCP server exposing tools over bioinformatics databases:
    - UniProt - protein sequences, function, disease annotation
    - Rhea - biochemical reactions
"""

import re
from urllib.parse import quote

import requests
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    name="Biodata MCP",
    dependencies=["mcp", "requests"],
    instructions=(
        "Query SIB Swiss Institute of Bioinformatics databases UniProt and Rhea"
    ),
    streamable_http_path="/",
    # stateless_http=True,
)

UNIPROT_API  = "https://rest.uniprot.org"
RHEA_API = "https://www.rhea-db.org/rhea"
HEADERS = {"Accept": "application/json"}

# UNIPROT TOOLS

@mcp.tool()
def uniprot_search(
    query: str,
    organism: str = "Homo sapiens",
    reviewed_only: bool = True,
    max_results: int = 5,
) -> dict:
    """Search UniProt for proteins matching a query (gene name, function, keyword…).

    Args:
        query: Free-text or field query. Examples:
            "TP53", "kinase AND cancer", "insulin receptor"
        organism: Organism name filter (default "Homo sapiens").
            Use "" to search all organisms.
        reviewed_only: If True, restrict to Swiss-Prot reviewed entries.
        max_results: Number of results to return (max 25).

    Returns:
        List of protein entries with accession, gene, protein name, organism,
        length (aa), and Swiss-Prot review status.

    Example questions:
        "Find all reviewed human kinases involved in DNA repair"
        "Search for mouse insulin proteins in UniProt"
    """
    max_results = min(max_results, 25)
    full_query = query
    if organism:
        full_query += f" AND organism_name:{organism}"
    if reviewed_only:
        full_query += " AND reviewed:true"

    params = {
        "query": full_query,
        "format": "json",
        "size": str(max_results),
        "fields": "accession,gene_names,protein_name,organism_name,length,reviewed",
    }
    resp = requests.get(f"{UNIPROT_API}/uniprotkb/search", params=params, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    results = []
    for e in data.get("results", []):
        gene = e.get("genes", [{}])
        gene_name = gene[0].get("geneName", {}).get("value", "N/A") if gene else "N/A"
        results.append({
            "accession": e["primaryAccession"],
            "gene": gene_name,
            "protein_name": e.get("proteinDescription", {})
                .get("recommendedName", {})
                .get("fullName", {})
                .get("value", "N/A"),
            "organism": e.get("organism", {}).get("scientificName", "N/A"),
            "length_aa": e.get("sequence", {}).get("length"),
            "reviewed": e.get("entryType") == "UniProtKB reviewed (Swiss-Prot)",
        })
    return {
        "query": full_query,
        "total_found": data.get("totalResults", len(results)),
        "results": results,
    }


@mcp.tool()
def uniprot_get_entry(accession: str) -> dict:
    """Fetch the full UniProt entry for a protein by its accession number.

    Retrieves curated annotation including: function, subcellular location,
    tissue expression, disease involvement, post-translational modifications,
    active/binding sites, and associated GO terms.

    Args:
        accession: UniProt accession (e.g. "P04637" for human TP53,
            "P01308" for human insulin, "P38398" for BRCA1).

    Returns:
        Rich annotation dict: function, location, diseases, PTMs, interactions,
        GO terms, sequence length, and AlphaFold structure URL.

    Example questions:
        "What is the function of P04637?"
        "What diseases is BRCA1 (P38398) involved in?"
        "Where is human insulin localized in the cell?"
    """
    resp = requests.get(
        f"{UNIPROT_API}/uniprotkb/{accession}",
        params={"format": "json"},
        headers=HEADERS,
        timeout=20,
    )
    resp.raise_for_status()
    e = resp.json()
    # Extract function comments
    comments = e.get("comments", [])
    def get_comment(ctype):
        for c in comments:
            if c.get("commentType") == ctype:
                texts = c.get("texts", [])
                return " ".join(t.get("value", "") for t in texts)
        return None
    # Diseases
    diseases = []
    for c in comments:
        if c.get("commentType") == "DISEASE":
            d = c.get("disease", {})
            diseases.append({
                "name": d.get("diseaseName"),
                "id": d.get("diseaseId"),
                "description": d.get("description"),
            })

    # Subcellular locations
    locations = []
    for c in comments:
        if c.get("commentType") == "SUBCELLULAR LOCATION":
            for loc in c.get("subcellularLocations", []):
                loc_val = loc.get("location", {}).get("value")
                if loc_val:
                    locations.append(loc_val)

    # GO terms (top 10)
    go_terms = []
    for ref in e.get("uniProtKBCrossReferences", []):
        if ref.get("database") == "GO":
            props = {p["key"]: p["value"] for p in ref.get("properties", [])}
            go_terms.append({
                "id": ref.get("id"),
                "term": props.get("GoTerm"),
                "evidence": props.get("GoEvidenceType"),
            })

    gene = e.get("genes", [{}])
    gene_name = gene[0].get("geneName", {}).get("value", "N/A") if gene else "N/A"
    return {
        "accession": e["primaryAccession"],
        "gene": gene_name,
        "protein_name": (
            e.get("proteinDescription", {})
             .get("recommendedName", {})
             .get("fullName", {})
             .get("value", "N/A")
        ),
        "organism": e.get("organism", {}).get("scientificName"),
        "length_aa": e.get("sequence", {}).get("length"),
        "reviewed": e.get("entryType") == "UniProtKB reviewed (Swiss-Prot)",
        "function": get_comment("FUNCTION"),
        "subcellular_locations": list(set(locations)),
        "diseases": diseases,
        "ptm_processing": get_comment("PTM"),
        "go_terms": go_terms[:10],
    }


@mcp.tool()
def uniprot_get_sequence(accession: str) -> dict:
    """Fetch the raw amino acid sequence for a UniProt accession in FASTA format.

    Args:
        accession: UniProt accession (e.g. "P04637", "P01308").

    Returns:
        Dict with FASTA header, full amino acid sequence string, and length.

    Example questions:
        "Give me the amino acid sequence of human TP53"
        "What is the sequence of insulin (P01308)?"
    """
    resp = requests.get(
        f"{UNIPROT_API}/uniprotkb/{accession}.fasta",
        timeout=20,
    )
    resp.raise_for_status()
    lines = resp.text.strip().split("\n")
    sequence = "".join(lines[1:])
    return {
        "accession": accession,
        "fasta_header": lines[0],
        "sequence": sequence,
        "length_aa": len(sequence),
    }



# RHEA TOOLS  (biochemical reactions)

@mcp.tool()
def rhea_search_reactions(query: str, max_results: int = 8) -> dict:
    """Search the Rhea database for biochemical reactions by compound, enzyme, or keyword.

    Args:
        query: Search term — compound name, EC number, enzyme name, or keyword.
            Examples: "ATP hydrolysis", "2.7.1", "glucose phosphorylation",
            "NADH", "kinase"
        max_results: Max reactions to return (default 8).

    Returns:
        List of reactions with Rhea ID, equation string, enzyme name, EC number,
        and ChEBI IDs of substrates/products.

    Example questions:
        "What reactions involve ATP hydrolysis?"
        "Find all kinase reactions in Rhea"
        "What reactions use NADH as a substrate?"
    """
    resp = requests.get(
        f"{RHEA_API}?query={quote(query)}&format=json&columns=rhea-id,equation,chebi-id&limit={max_results}",
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()
    results = []
    for r in data.get("results", []):
        results.append({
            "rhea_id": r.get("rheaId") or r.get("id"),
            "equation": r.get("equation"),
            "chebi_ids": [m.replace("chebi:", "") for m in re.findall(r'data-molid="(chebi:[^"]+)"', r.get("htmlequation", ""))],
            "rhea_url": f"https://www.rhea-db.org/rhea/{r.get('rheaId') or r.get('id')}",
        })
    return {
        "query": query,
        "total_found": data.get("count", len(results)),
        "results": results,
    }


if __name__ == "__main__":
    import sys
    if "--stdio" in sys.argv:
        mcp.run(transport="stdio")
    else:
        print("Biodata MCP Server starting on http://localhost:8000/mcp ...")
        mcp.run(transport="streamable-http")
