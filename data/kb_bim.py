"""
Ground-truth knowledge base for BIM / digital construction terminology.

Every synthetic training example is derived from this file. Keeping the
facts hand-curated here (instead of asking an LLM to invent Q&A pairs from
scratch) means the ground truth is auditable and the eval set can be graded
objectively against it.

Each entry:
  id        short slug, used for cross-references in `related`
  term      canonical term
  acronym   acronym form if one exists, else None
  category  coarse grouping, used to vary phrasing/answers
  definition  1-3 sentence definition
  example   1-2 sentence real-world usage/example
  related   list of other entry ids worth contrasting this term with
"""

KB = [
    dict(id="bim", term="Building Information Modeling", acronym="BIM", category="core",
         definition="A process for creating and managing information about a built asset across its "
                     "lifecycle, centered on a digital model that combines geometry with structured data "
                     "(materials, cost, schedule, performance) rather than just drawings.",
         example="A design team uses a single coordinated 3D model, instead of separate 2D drawing sets, "
                  "so that architectural, structural, and MEP information stay consistent as the design changes.",
         related=["ifc", "cde", "lod"]),

    dict(id="3d-bim", term="3D BIM", acronym=None, category="dimension",
         definition="The base geometric dimension of BIM: a coordinated three-dimensional digital model of "
                     "the physical building elements.",
         example="Architects, structural engineers, and MEP designers each contribute a 3D model that is "
                  "federated into one combined model for coordination.",
         related=["4d-bim", "bim"]),

    dict(id="4d-bim", term="4D BIM", acronym=None, category="dimension",
         definition="BIM enriched with the fourth dimension, time: linking model elements to a construction "
                     "schedule so sequencing can be simulated and visualized.",
         example="A contractor links the structural steel model elements to the project schedule to produce "
                  "a time-lapse simulation of the erection sequence, catching sequencing clashes before they occur on site.",
         related=["3d-bim", "5d-bim"]),

    dict(id="5d-bim", term="5D BIM", acronym=None, category="dimension",
         definition="BIM enriched with the fifth dimension, cost: linking model elements to quantities and "
                     "cost data so that estimates update automatically as the design changes.",
         example="A quantity surveyor extracts material quantities directly from the model to generate an "
                  "up-to-date cost estimate whenever the design is revised, instead of manually re-measuring drawings.",
         related=["4d-bim", "quantity-takeoff"]),

    dict(id="6d-bim", term="6D BIM", acronym=None, category="dimension",
         definition="BIM extended to cover sustainability and energy performance analysis, using the model "
                     "to simulate energy consumption and environmental impact.",
         example="A design team runs an energy simulation directly against the BIM model to compare glazing "
                     "options before construction documents are finalized.",
         related=["5d-bim", "7d-bim"]),

    dict(id="7d-bim", term="7D BIM", acronym=None, category="dimension",
         definition="BIM extended into facility management: the as-built model carries asset data (warranties, "
                     "maintenance schedules, manuals) used by the owner to operate the building after handover.",
         example="A facilities manager looks up a rooftop unit's warranty and maintenance history directly in "
                  "the handover model instead of searching paper O&M binders.",
         related=["6d-bim", "cobie"]),

    dict(id="lod", term="Level of Development", acronym="LOD", category="standard",
         definition="A scale (commonly 100 through 500) describing how much a model element can be relied "
                     "on for a given purpose, combining both its geometric detail and the reliability of its "
                     "attached information at that stage of the project.",
         example="A wall might be LOD 200 (approximate size, generic material) during schematic design and "
                     "LOD 400 (exact dimensions, fabrication-ready) once it's ready for the shop drawing stage.",
         related=["loi", "bim"]),

    dict(id="loi", term="Level of Information", acronym="LOI", category="standard",
         definition="The non-graphical, data side of model maturity: how complete and reliable the attribute "
                     "data (specifications, performance data, classification) attached to an element is, as "
                     "distinct from its geometric detail.",
         example="A door object might have full geometric detail but still be missing its fire rating and "
                     "hardware schedule data, meaning its LOI is lower than its geometric LOD would suggest.",
         related=["lod"]),

    dict(id="ifc", term="Industry Foundation Classes", acronym="IFC", category="standard",
         definition="An open, vendor-neutral data schema maintained by buildingSMART for describing building "
                     "and infrastructure data, used to exchange BIM models between different software packages.",
         example="A structural model authored in Tekla Structures is exported as an IFC file so it can be "
                     "opened and coordinated against an architectural model authored in Revit.",
         related=["openbim", "bcf"]),

    dict(id="openbim", term="openBIM", acronym=None, category="standard",
         definition="A vendor-neutral, collaborative approach to BIM built on open standards such as IFC and "
                     "BCF, so that project data isn't locked into one software vendor's proprietary format.",
         example="A project team mandates IFC deliverables at each milestone so that any team member can open "
                     "and check the model regardless of which authoring tool produced it.",
         related=["ifc", "bcf"]),

    dict(id="bcf", term="BIM Collaboration Format", acronym="BCF", category="standard",
         definition="An open file format used to communicate model-based issues (like clashes) between "
                     "different BIM software, carrying a viewpoint, screenshot, and comments without needing "
                     "to exchange the whole model.",
         example="A coordinator flags a clash between a duct and a beam in Navisworks and exports it as a BCF "
                     "file so the MEP engineer can open the exact same viewpoint directly in Revit.",
         related=["ifc", "clash-detection"]),

    dict(id="cde", term="Common Data Environment", acronym="CDE", category="process",
         definition="A single agreed source of information for a project, used to collect, manage, and "
                     "disseminate documents and models between all members of the project team through a "
                     "managed workflow of states.",
         example="Instead of emailing model files around, the whole project team uploads and retrieves "
                     "current information from a shared platform such as BIM 360 or Autodesk Construction Cloud.",
         related=["cde-states", "bim"]),

    dict(id="cde-states", term="CDE workflow states", acronym=None, category="process",
         definition="Under ISO 19650, information in the CDE moves through four states: Work in Progress "
                     "(not yet shared), Shared (checked and available to other teams), Published (authorized "
                     "for a specific use like construction or tender), and Archive (a fixed record).",
         example="A structural model stays in the Work in Progress area until it passes an internal QA check, "
                     "at which point it moves to Shared so the architecture team can coordinate against it.",
         related=["cde", "iso19650"]),

    dict(id="iso19650", term="ISO 19650", acronym=None, category="standard",
         definition="An international standard for managing information over the whole lifecycle of a built "
                     "asset using BIM, covering concepts and principles (Part 1), the delivery phase "
                     "(Part 2), the operational phase (Part 3), and security-minded information management "
                     "(Part 5).",
         example="A client references ISO 19650-2 in tender documents to require bidders to submit a BIM "
                     "Execution Plan describing how they'll manage project information.",
         related=["bep", "eir", "cde-states"]),

    dict(id="eir", term="Exchange Information Requirements", acronym="EIR", category="document",
         definition="A document issued by the appointing party (client) stating what information it needs, "
                     "in what format, and at what stage, which bidders respond to when preparing their BIM "
                     "Execution Plan.",
         example="A hospital client's EIR specifies that all fire doors must be modeled to LOD 400 and "
                     "classified using Uniclass so the data can feed the facility management system at handover.",
         related=["bep", "iso19650"]),

    dict(id="bep", term="BIM Execution Plan", acronym="BEP", category="document",
         definition="A document produced by the delivery team explaining how the information management "
                     "aspects of a project will be carried out — roles, software, model breakdown, LOD "
                     "targets, and CDE workflow — in response to the client's Exchange Information Requirements.",
         example="Before design starts, the design team submits a pre-appointment BEP showing it has the "
                     "capability and resources to meet the client's EIR, then a detailed post-appointment BEP once awarded the work.",
         related=["eir", "midp"]),

    dict(id="midp", term="Master Information Delivery Plan", acronym="MIDP", category="document",
         definition="A schedule, assembled from each task team's Task Information Delivery Plan, showing when "
                     "project information (models, drawings, schedules) will be prepared, by whom, and using "
                     "which resources, across the whole project.",
         example="The MIDP shows that the structural model deliverable for the tender stage is due four weeks "
                     "before the architectural coordination deliverable, so the schedule can be checked for feasibility.",
         related=["tidp", "bep"]),

    dict(id="tidp", term="Task Information Delivery Plan", acronym="TIDP", category="document",
         definition="A schedule produced by an individual task team (e.g. the structural engineers) listing "
                     "the specific information deliverables it's responsible for and when each is due; TIDPs "
                     "are combined into the project-wide Master Information Delivery Plan.",
         example="The MEP task team's TIDP lists the ductwork model, equipment schedule, and riser diagrams it "
                     "owes at each project stage.",
         related=["midp"]),

    dict(id="cobie", term="Construction Operations Building Information Exchange", acronym="COBie", category="standard",
         definition="A structured, spreadsheet-friendly data format for handing over non-graphical asset "
                     "information — equipment lists, warranties, spare parts, maintenance schedules — from "
                     "the construction team to the building owner's facility management system.",
         example="At handover, the contractor delivers a COBie spreadsheet listing every HVAC unit with its "
                     "model number, installation date, and warranty period so facilities staff can load it into their CMMS.",
         related=["7d-bim", "ifc"]),

    dict(id="clash-detection", term="Clash Detection", acronym=None, category="process",
         definition="The process of automatically checking a federated model for physical conflicts between "
                     "elements from different disciplines (e.g. a duct routed through a structural beam) "
                     "before construction, typically using software such as Navisworks or Solibri.",
         example="Before issuing for construction, the coordination team runs a clash detection pass and finds "
                     "that a sprinkler main clashes with a structural beam, so the MEP model is rerouted before it reaches the site.",
         related=["model-federation", "bcf"]),

    dict(id="model-federation", term="Model Federation", acronym=None, category="process",
         definition="Combining multiple discipline-specific models (architectural, structural, MEP) into one "
                     "linked, coordinated view for clash checking and review, without merging them into a "
                     "single authored file.",
         example="A coordination model links the architectural, structural, and MEP Revit models via shared "
                     "coordinates so the whole team can review them together in Navisworks.",
         related=["clash-detection", "cde"]),

    dict(id="quantity-takeoff", term="Quantity Takeoff", acronym=None, category="process",
         definition="Extracting material and component quantities (concrete volume, wall area, number of "
                     "doors) directly from a BIM model, used to support cost estimating (5D BIM).",
         example="An estimator pulls a live concrete volume quantity from the structural model instead of "
                     "manually measuring formwork drawings.",
         related=["5d-bim"]),

    dict(id="digital-twin", term="Digital Twin", acronym=None, category="process",
         definition="A live, continuously updated digital representation of a physical asset, synchronized "
                     "with real-world sensor or operational data, going beyond a static as-built BIM model.",
         example="A building owner connects real-time sensor data from HVAC equipment to the handover BIM "
                     "model, turning it into a digital twin that flags underperforming equipment automatically.",
         related=["7d-bim", "scan-to-bim"]),

    dict(id="scan-to-bim", term="Scan-to-BIM", acronym=None, category="process",
         definition="Capturing an existing building's geometry using laser scanning (producing a point cloud) "
                     "and then modeling it into a BIM model, used for renovation or retrofit projects where no "
                     "accurate as-built model exists.",
         example="Before designing a retrofit, a survey team laser-scans an existing 1960s office building and "
                     "the point cloud is used to author an accurate as-built Revit model.",
         related=["point-cloud", "digital-twin"]),

    dict(id="point-cloud", term="Point Cloud", acronym=None, category="process",
         definition="A large set of XYZ coordinate points, typically captured by laser scanning or "
                     "photogrammetry, representing the surfaces of a physical space or object.",
         example="A laser scanner captures millions of points describing an existing plant room, which is then "
                     "used as a reference to trace new pipework in the BIM model.",
         related=["scan-to-bim"]),

    dict(id="dfma", term="Design for Manufacture and Assembly", acronym="DfMA", category="process",
         definition="A design approach that plans building components to be manufactured off-site and "
                     "assembled on-site, using the BIM model to drive fabrication and coordinate the assembly sequence.",
         example="A contractor uses DfMA to have bathroom pods manufactured in a factory from the coordinated "
                     "BIM model and craned into place on site, cutting on-site trade work.",
         related=["4d-bim", "clash-detection"]),

    dict(id="bim-manager", term="BIM Manager", acronym=None, category="role",
         definition="The person responsible for setting and enforcing BIM standards, workflows, and software "
                     "configuration across an organization or project, typically at a strategic/organizational level.",
         example="The BIM Manager defines the company's model naming convention and template standards that "
                     "every project must follow.",
         related=["bim-coordinator", "information-manager"]),

    dict(id="bim-coordinator", term="BIM Coordinator", acronym=None, category="role",
         definition="The person responsible for day-to-day model coordination on a specific project — running "
                     "clash detection, managing the federated model, and tracking issue resolution between disciplines.",
         example="The BIM Coordinator runs the weekly clash detection meeting, walking the design team through "
                     "unresolved clashes flagged in Navisworks.",
         related=["bim-manager", "clash-detection"]),

    dict(id="information-manager", term="Information Manager", acronym=None, category="role",
         definition="The ISO 19650 role responsible for managing the information management process on a "
                     "project — setting up and administering the CDE, standards, and workflows described in the BEP.",
         example="The Information Manager confirms that each task team's model meets the agreed file naming "
                     "and LOD requirements before it's allowed to move from Work in Progress to Shared in the CDE.",
         related=["bim-manager", "cde"]),

    dict(id="revit", term="Revit", acronym=None, category="software",
         definition="Autodesk's BIM authoring software, widely used for architectural, structural, and MEP "
                     "modeling, built around parametric elements and family-based components.",
         example="An architect models a building's walls, doors, and windows as parametric families in Revit "
                     "so that changing a wall type updates every instance of it automatically.",
         related=["navisworks", "bim"]),

    dict(id="navisworks", term="Navisworks", acronym=None, category="software",
         definition="Autodesk's model review and coordination software, used to combine (federate) models "
                     "from multiple authoring tools for clash detection, 4D simulation, and walkthroughs.",
         example="The coordination team imports the Revit and Tekla models into Navisworks to run a clash "
                     "detection test between structural steel and ductwork.",
         related=["clash-detection", "revit"]),

    dict(id="solibri", term="Solibri", acronym=None, category="software",
         definition="A model-checking software used to validate BIM models against rules — such as clash "
                     "detection, code compliance, and information/data quality checks — typically working from IFC files.",
         example="A team runs an IFC export through Solibri to check that every room has been classified "
                     "correctly and that fire escape routes meet minimum width rules.",
         related=["ifc", "clash-detection"]),

    dict(id="acc", term="Autodesk Construction Cloud", acronym="ACC", category="software",
         definition="Autodesk's cloud-based common data environment platform for managing project documents, "
                     "models, issues, and workflows across design and construction teams.",
         example="The general contractor manages RFIs, submittals, and the current model set through Autodesk "
                     "Construction Cloud so every subcontractor works from the same information.",
         related=["cde", "bim360"]),

    dict(id="bim360", term="BIM 360", acronym=None, category="software",
         definition="Autodesk's earlier cloud collaboration platform (predecessor to Autodesk Construction "
                     "Cloud) used as a Common Data Environment for design and construction teams.",
         example="A project that started before the platform migration still manages its document approvals "
                     "in BIM 360 rather than the newer Autodesk Construction Cloud.",
         related=["acc", "cde"]),

    dict(id="rfi-bim", term="RFI in a BIM Workflow", acronym="RFI", category="process",
         definition="A Request for Information raised on site or in design review; in a BIM-enabled project "
                     "an RFI is often linked directly to the specific model element and issue viewpoint "
                     "(e.g. via BCF) rather than a text description alone.",
         example="A site superintendent raises an RFI about a beam-to-column connection detail and attaches "
                     "a BCF viewpoint from the federated model so the structural engineer can see exactly which connection is in question.",
         related=["bcf", "clash-detection"]),

    dict(id="uniclass", term="Uniclass", acronym=None, category="standard",
         definition="A unified classification system for the UK construction industry, used to consistently "
                     "classify and tag objects, systems, and spaces in a BIM model (and in COBie data) so "
                     "information can be organized and retrieved reliably.",
         example="Every door object in the model is tagged with a Uniclass product code so the specification "
                     "and the model stay linked and searchable.",
         related=["cobie", "ifc"]),

    dict(id="asset-information-model", term="Asset Information Model", acronym="AIM", category="document",
         definition="Under ISO 19650, the information model maintained by the owner/operator during the "
                     "operational phase of an asset, built from the Project Information Model handed over at "
                     "the end of construction.",
         example="After handover, the facilities team maintains the Asset Information Model, updating it as "
                     "equipment is replaced or refurbished during the building's operational life.",
         related=["project-information-model", "cobie"]),

    dict(id="project-information-model", term="Project Information Model", acronym="PIM", category="document",
         definition="Under ISO 19650, the information model developed and maintained during the design and "
                     "construction phase of a project, which becomes the basis for the Asset Information "
                     "Model handed over to the owner at completion.",
         example="Throughout construction, the design and construction teams keep updating the Project "
                     "Information Model, which is then handed over to become the client's Asset Information Model.",
         related=["asset-information-model", "iso19650"]),

    dict(id="worksharing", term="Worksharing", acronym=None, category="process",
         definition="A workflow, built into authoring tools like Revit, that lets multiple team members edit "
                     "the same central model simultaneously by checking out and synchronizing local copies.",
         example="Three architects work on the same Revit model at once, each owning different areas of the "
                     "building, synchronizing their local copies back to the central model every hour.",
         related=["revit", "cde"]),

    dict(id="model-checking", term="Model Checking", acronym=None, category="process",
         definition="Automated or manual validation of a BIM model against a set of rules — geometric, "
                     "informational, or regulatory — to catch errors before the model is relied on for "
                     "coordination or construction.",
         example="Before a model is published to the CDE, an automated model check confirms every element has "
                     "the required classification and parameter data filled in.",
         related=["solibri", "loi"]),

    dict(id="federated-model", term="Federated Model", acronym=None, category="process",
         definition="A composite model formed by linking together separate discipline models (architecture, "
                     "structure, MEP) for coordination purposes, without combining them into one authored file.",
         example="The federated model used in the weekly coordination meeting links the architectural, "
                     "structural, and MEP Revit models by shared coordinates.",
         related=["model-federation", "clash-detection"]),

    dict(id="bim-dictionary", term="BIM Dictionary / Data Dictionary", acronym=None, category="standard",
         definition="A structured reference defining the properties, classifications, and terminology used "
                     "consistently across a project's model data, so different teams use the same meaning for "
                     "the same term or property.",
         example="The project's data dictionary defines exactly what 'fire rating' means and what units it's "
                     "recorded in, so every discipline enters it consistently.",
         related=["uniclass", "loi"]),

    dict(id="appointing-party", term="Appointing Party", acronym=None, category="role",
         definition="The ISO 19650 term for the client or organization commissioning the project, responsible "
                     "for issuing the Exchange Information Requirements.",
         example="The hospital trust, as the appointing party, issues an EIR requiring BIM Level 2 compliance "
                     "from all bidders.",
         related=["eir", "lead-appointed-party"]),

    dict(id="lead-appointed-party", term="Lead Appointed Party", acronym=None, category="role",
         definition="The ISO 19650 term for the main delivery team (e.g. the lead designer or main "
                     "contractor) appointed by the appointing party, responsible for coordinating the task "
                     "teams below it and submitting the BIM Execution Plan.",
         example="The main contractor, as lead appointed party, coordinates the structural, MEP, and "
                     "facade subcontractors' task teams and consolidates their TIDPs into the project MIDP.",
         related=["appointing-party", "bep"]),
]

KB_BY_ID = {entry["id"]: entry for entry in KB}
