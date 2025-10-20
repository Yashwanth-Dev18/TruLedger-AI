import csv


job_list = {
    "Business & Management": [
        "Retail merchandiser", "Catering manager", "Product manager", "Sports administrator", 
        "Hotel manager", "Health service manager", "Retail manager", "Chief Marketing Officer",
        "Public relations officer", "Magazine features editor", "Politician's assistant",
        "Visual merchandiser", "Administrator - education", "Risk analyst", "Advertising account planner",
        "Financial adviser", "Civil Service fast streamer", "Barista", "Trade mark attorney",
        "Engineer - technical sales", "Tourist information centre manager", "Location manager",
        "Research officer - trade union", "Air cabin crew", "Chief of Staff", "Merchandiser - retail",
        "Press sub", "Management consultant", "Investment banker - operational", "Operational investment banker",
        "Buyer - industrial", "Armed forces logistics/support/administrative officer", "Commissioning editor",
        "Records manager", "Public relations account executive", "Personnel officer",
        "Sales promotion account executive", "Pension scheme manager", "Regulatory affairs officer",
        "Futures trader", "Sales executive", "Education administrator", "Civil Service administrator",
        "Copywriter - advertising", "Warehouse manager", "Freight forwarder", "Research officer - political party",
        "Logistics and distribution manager", "Chief Strategy Officer", "Public house manager",
        "Fitness centre manager", "Medical sales representative", "Air broker", "Dealer",
        "Training and development officer", "Human resources officer", "Careers information officer",
        "Claims inspector/assessor", "Advertising copywriter", "Radio broadcast assistant",
        "Administrator - arts", "Tax inspector", "Loss adjuster - chartered", "Chartered loss adjuster",
        "Sports development officer", "Event organiser", "Administrator - local government",
        "Producer - radio", "Advertising account executive", "Leisure centre manager",
        "Editor - magazine features", "Administrator - charities/voluntary organisations",
        "Press photographer", "Journalist - newspaper", "Purchasing manager", "Production manager",
        "Bookseller", "Medical secretary", "Buyer - retail", "Company secretary",
        "Trading standards officer", "Marketing executive", "Market researcher", "Administrator",
        "Chief Financial Officer", "Production assistant - radio", "Sales professional - IT",
        "Media planner", "Call centre manager", "Chief Executive Officer", "Media buyer",
        "Retail buyer", "Insurance claims handler", "Facilities manager", "Chief Operating Officer",
        "Secretary/administrator", "Information systems manager", "Industrial buyer",
        "Public affairs consultant", "Quarry manager"
    ],
    "Healthcare & Medical": [
        "Occupational hygienist", "Orthoptist", "Physiotherapist", "Nurse - children's",
        "Neurosurgeon", "Clinical psychologist", "Biochemist - clinical", "Health promotion specialist",
        "Biomedical scientist", "Herbalist", "Child psychotherapist", "Occupational psychologist",
        "Pharmacologist", "Hospital pharmacist", "Counselling psychologist", "Acupuncturist",
        "Forensic psychologist", "Dispensing optician", "Homeopath", "Pathologist", "Chiropodist",
        "Learning disability nurse", "Community pharmacist", "Radiographer - therapeutic",
        "Doctor - hospital", "Dance movement psychotherapist", "Occupational therapist",
        "Mental health nurse", "Therapist - horticultural", "Therapist - sports",
        "Scientist - clinical (histocompatibility and immunogenetics)", "Health physicist",
        "Nutritional therapist", "Ambulance person", "Medical physicist", "Therapist - occupational",
        "Podiatrist", "Diagnostic radiographer", "Horticultural therapist", "Therapist - drama",
        "Physiological scientist", "Medical technical officer", "Clinical cytogeneticist",
        "Surgeon", "Immunologist", "Psychotherapist", "Psychologist - sport and exercise",
        "Psychologist - forensic", "Clinical research associate", "Scientist - audiological",
        "Embryologist - clinical", "Optometrist", "Therapist - music", "Scientist - physiological",
        "Psychologist - counselling", "Exercise physiologist", "Pharmacist - community",
        "Cytogeneticist", "Educational psychologist", "Paramedic", "Barrister", "Oncologist",
        "Psychiatric nurse", "Veterinary surgeon", "Phytotherapist", "Toxicologist", "Nurse - mental health",
        "Radiographer - diagnostic", "Hospital doctor", "Psychologist - clinical", "Psychotherapist - child",
        "Paediatric nurse", "Clinical biochemist", "Physicist - medical", "Sport and exercise psychologist",
        "Optician - dispensing", "Music therapist", "Audiological scientist", "Health visitor",
        "Doctor - general practice", "General practice doctor", "Pharmacist - hospital",
        "Research scientist (medical)", "Osteopath", "Environmental health practitioner",
        "Psychiatrist", "Scientist - biomedical"
    ],
    "Creative Arts & Media": [
        "Presenter - broadcasting", "Stage manager", "Theatre director", "Production assistant - television",
        "Editor - film/video", "Programmer - multimedia", "Television production assistant", "Musician",
        "Illustrator", "Fine artist", "Interior and spatial designer", "Animator", "Textile designer",
        "Curator", "Designer - ceramics/pottery", "Copy", "Make", "Television camera operator",
        "Art gallery manager", "Video editor", "Broadcast journalist", "Magazine journalist",
        "Glass blower/designer", "Exhibitions officer - museum/gallery", "Music tutor", "Writer",
        "Web designer", "Designer - multimedia", "Producer - television/film/video", "Jewellery designer",
        "Designer - furniture", "Museum/gallery exhibitions officer", "Broadcast presenter",
        "Product designer", "Industrial/product designer", "Education officer - museum",
        "Theatre manager", "Editor - commissioning", "Television/film/video producer",
        "Museum/gallery conservator", "Gaffer", "Film/video editor", "Heritage manager", "Artist",
        "Dancer", "Designer - jewellery", "Conservator - furniture", "Exhibition designer",
        "Designer - industrial/product", "Designer - exhibition/display", "Programme researcher - broadcasting/film/video",
        "Camera operator", "Archivist", "Public librarian", "Interpreter", "Science writer",
        "Designer - television/film set", "Special effects artist", "Therapist - art",
        "Conservator - museum/gallery", "Multimedia programmer", "Designer - interior/spatial",
        "Ceramics designer", "Designer - textile", "Set designer", "Furniture conservator/restorer",
        "Furniture designer", "Television floor manager", "Radio producer"
    ],
    "Engineering & Technology": [
        "Control and instrumentation engineer", "Engineer - electronics", "Engineer - manufacturing",
        "Engineer - communications", "Communications engineer", "Materials engineer", "Broadcast engineer",
        "Aeronautical engineer", "Applications developer", "Systems developer", "Network engineer",
        "Telecommunications researcher", "Armed forces technical officer", "Petroleum engineer",
        "Engineer - control and instrumentation", "Water engineer", "Engineer - agricultural",
        "Engineer - mining", "Metallurgist", "Mining engineer", "Architectural technologist",
        "Engineer - civil (contracting)", "Engineer - water", "Electronics engineer",
        "Engineer - civil (consulting)", "Naval architect", "Manufacturing engineer",
        "Engineer - aeronautical", "Chemical engineer", "Engineering geologist", "Engineer - petroleum",
        "IT consultant", "Engineer - maintenance", "Engineer - automotive", "Engineer - building services",
        "Engineer - site", "Software engineer", "Contracting civil engineer", "Electrical engineer",
        "Biomedical engineer", "Database administrator", "Building services engineer",
        "Manufacturing systems engineer", "Maintenance engineer", "Site engineer", "Engineer - materials",
        "Engineer - structural", "Programmer - applications", "Production engineer",
        "Engineer - broadcasting (operations)", "Technical brewer", "Structural engineer",
        "Mechanical engineer", "Engineer - drilling", "IT trainer", "Energy engineer", "Systems analyst",
        "Data scientist", "Engineer - production", "Engineer - land", "Chief Technology Officer",
        "Drilling engineer", "Engineer - biomedical", "Mudlogger", "Geoscientist", "Seismic interpreter",
        "Clothing/textile technologist", "Colour technologist", "Garment/textile technologist"
    ],
    "Education & Teaching": [
        "English as a second language teacher", "Primary school teacher", "Early years teacher",
        "Teacher - primary school", "Teacher - special educational needs", "Teacher - English as a foreign language",
        "TEFL teacher", "Lecturer - further education", "English as a foreign language teacher",
        "Associate Professor", "Museum education officer", "Secondary school teacher",
        "Teacher - early years/pre", "Teacher - adult education", "Higher education careers adviser",
        "Armed forces training and education officer", "Special educational needs teacher",
        "Private music teacher", "Learning mentor", "Teacher - secondary school", "Lecturer - higher education",
        "Community education officer", "Outdoor activities/education manager", "Teaching laboratory technician",
        "Further education lecturer", "Professor Emeritus", "Careers adviser"
    ],
    "Science & Research": [
        "Geologist - engineering", "Hydrogeologist", "Geophysicist/field seismologist", "Herpetologist",
        "Wellsite geologist", "Operational researcher", "Archaeologist", "Plant breeder/geneticist",
        "Ecologist", "Lexicographer", "Analytical chemist", "Environmental consultant", "Scientist - marine",
        "Product/process development scientist", "Water quality scientist", "Animal nutritionist",
        "Statistician", "Scientific laboratory technician", "Chemist - analytical", "Geneticist - molecular",
        "Oceanographer", "Field trials officer", "Scientist - research (physical sciences)", "Field seismologist",
        "Geologist - wellsite", "Hydrologist", "Operations geologist", "Scientist - research (medical)",
        "Animal technologist", "Social researcher", "Geochemist", "Research scientist (physical sciences)",
        "Research scientist (maths)", "Scientist - research (maths)", "Soil scientist",
        "Research scientist (life sciences)", "Economist"
    ],
    "Legal & Government": [
        "Police officer", "Lawyer", "Patent attorney", "Chartered legal executive (England and Wales)",
        "Emergency planning/management officer", "Fisheries officer", "Immigration officer", "Prison officer",
        "Intelligence analyst", "Social research officer - government", "Solicitor - Scotland",
        "Probation officer", "Licensed conveyancer", "Equality and diversity officer", "Local government officer",
        "Barrister's clerk", "Solicitor"
    ],
    "Finance & Accounting": [
        "Chartered public finance accountant", "Corporate investment banker", "Accounting technician",
        "Tax adviser", "Accountant - chartered public finance", "Accountant - chartered certified",
        "Senior tax professional/tax inspector", "Financial trader", "Pensions consultant",
        "Equities trader", "Comptroller", "Retail banker", "Insurance underwriter", "Insurance broker",
        "Accountant - chartered", "Chartered accountant", "Investment banker - corporate",
        "Investment analyst", "Insurance risk surveyor"
    ],
    "Construction & Surveying": [
        "Quantity surveyor", "Land/geomatics surveyor", "Hydrographic surveyor", "Rural practice surveyor",
        "Historic buildings inspector/conservation officer", "Surveyor - land/geomatics",
        "Conservation officer - historic buildings", "Surveyor - rural practice", "Architect",
        "Surveyor - mining", "Surveyor - minerals", "Surveyor - hydrographic", "Building control surveyor",
        "Commercial/residential surveyor", "Town planner", "Building surveyor", "Planning and development surveyor",
        "Estate manager/land agent", "Landscape architect", "Minerals surveyor", "Cartographer"
    ],
    "Environment & Agriculture": [
        "Arboriculturist", "Horticultural consultant", "Tree surgeon", "Environmental manager",
        "Horticulturist - commercial", "Agricultural consultant", "Forest/woodland manager",
        "Amenity horticulturist", "Warden/ranger", "Farm manager", "Environmental education officer",
        "Nature conservation officer", "Energy manager", "Waste management officer", "Commercial horticulturist"
    ],
    "Social Services & Community": [
        "Counsellor", "Aid worker", "Education officer - community", "Volunteer coordinator", "Charity officer",
        "Arts development officer", "Advice worker", "Race relations officer", "Community arts worker",
        "Community development worker", "Development worker - international aid", "Charity fundraiser",
        "Development worker - community"
    ],
    "Transportation & Entertainment Services": [
        "Transport planner", "Ship broker", "Air traffic controller", "Tour manager", "Pilot - airline",
        "Airline pilot", "Cabin crew", "Theme park manager", "Tourism officer", "Travel agency manager",
        "Restaurant manager - fast food"
    ]
}

# Output CSV file name
output_file = "job_categories.csv"

# Get all category names (CSV headers)
categories = list(job_list.keys())

# Find the max number of jobs in any category
max_len = max(len(jobs) for jobs in job_list.values())

# Open CSV for writing
with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)

    # Write header row
    writer.writerow(categories)

    # Write each row of jobs
    for i in range(max_len):
        row = []
        for category in categories:
            # Add job name if exists, else blank cell
            row.append(job_list[category][i] if i < len(job_list[category]) else "")
        writer.writerow(row)

print(f"✅ CSV file '{output_file}' created successfully!")