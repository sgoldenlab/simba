# -*- coding: utf-8 -*-
"""Extract corpus statistics from the local SimBA use-case paper PDFs and write a
small, committable JSON that the docs renderer consumes.

The PDFs themselves are NOT in the repo (they live locally, e.g. F:\\simba_papers) and
cannot be reached by the CI/ReadTheDocs build. This script is the LOCAL step: run it
whenever the paper folder changes, then commit misc/corpus_stats.json. The renderer
(usecase_map_stats.py) reads only that JSON, so the docs build stays PDF-free.

If the paper folder is missing/empty (e.g. on another machine or in CI), the existing
committed JSON is kept untouched -- the build never breaks.

Run:  python misc/extract_corpus_stats.py  [pdf_dir]
"""
import os, re, sys, json, subprocess, collections
from datetime import date

PDF_DIR = sys.argv[1] if len(sys.argv) > 1 else r"F:\simba_papers"
OUT = os.path.join("misc", "corpus_stats.json")

# --- lexicons: category label -> regex of surface forms (word-boundary, lowercased) ---
# Counting is DOCUMENT frequency: each paper contributes at most once per label.
BEHAVIORS = {
    "Social interaction": r"social (?:interaction|behavior|behaviour|investigation|approach)|social contact",
    "Aggression / attack": r"aggress|attack|fighting|resident.?intruder|biting",
    "Grooming": r"groom",
    "Freezing": r"freezing",
    "Locomotion / open field": r"locomot|open.?field|ambulat|distance travel",
    "Rearing": r"\brear(?:ing|s|ed)?\b",
    "Sniffing": r"sniff",
    "Immobility (FST/TST)": r"immobil|tail suspension|forced swim",
    "Mating / sexual": r"\bmating\b|copulat|intromission|sexual behav|mounting behav",
    "Avoidance": r"avoidance",
    "Head-twitch (HTR)": r"head.?twitch|\bhtr\b",
    "Feeding / consumption": r"feeding|food intake|self.?administrat|consumption",
    "Gait / balance": r"\bgait\b|rotarod|balance beam|posture",
    "Pup retrieval / maternal": r"pup retrieval|maternal behav|nest building|nesting",
}
REGIONS = {
    "Amygdala": r"amygdala|\bbla\b|\bcea\b|\bmea\b",
    "Nucleus accumbens": r"nucleus accumbens|accumbens|\bnac\b",
    "Prefrontal cortex": r"prefrontal|\bpfc\b|\bmpfc\b",
    "Hippocampus": r"hippocamp",
    "BNST": r"bed nucleus of the stria|\bbnst\b",
    "Hypothalamus": r"hypothalam",
    "VTA": r"ventral tegmental|\bvta\b",
    "Striatum": r"striat",
    "Insula": r"insula",
    "Cingulate cortex": r"cingulate",
    "Thalamus": r"\bthalam",
    "Periaqueductal gray": r"periaqueductal|\bpag\b",
    "Habenula": r"habenula",
    "Cerebellum": r"cerebell",
}
METHODS = {
    "DeepLabCut": r"deeplabcut|\bdlc\b",
    "SLEAP": r"\bsleap\b",
    "Optogenetics": r"optogenetic|channelrhodopsin|\bchr2\b|halorhodopsin",
    "Fiber photometry": r"photometry|\bgcamp\b",
    "Chemogenetics (DREADD)": r"dreadd|chemogenetic|\bcno\b|hm[34]d",
    "Calcium imaging": r"calcium imaging|miniscope|two.?photon|2.?photon",
    "Electrophysiology": r"electrophysiolog|patch.?clamp|single.?unit|in vivo record",
    "Transcriptomics": r"rna.?seq|transcriptom|single.?cell rna",
    "SHAP explainability": r"\bshap\b|shapley",
    "BORIS annotation": r"\bboris\b",
    "EthoVision": r"ethovision",
}
DISEASES = {
    "Addiction / substance use": r"addiction|opioid|cocaine|fentanyl|oxycodone|alcohol|ethanol|methamphetamine|nicotine|psychostimulant",
    "Pain / analgesia": r"\bpain\b|analgesi|nocicep|hyperalgesi",
    "Stress / depression": r"chronic stress|social defeat|depress|anhedoni|despair",
    "Anxiety": r"anxiet|anxio",
    "Autism (ASD)": r"autism|\basd\b|autistic",
    "Parkinson's": r"parkinson|alpha.?synuclein|\bmptp\b|6.?ohda",
    "Alzheimer's / dementia": r"alzheimer|amyloid|\btau\b|dementia",
    "Epilepsy / seizure": r"epilep|seizure|convuls",
    "Psychedelics": r"psychedelic|psilocy|\blsd\b|ketamine|\bdmt\b|serotonergic",
}

# Curated scientific concept gazetteer for the word cloud -- so it shows real entities
# (regions, neuromodulators, drugs, paradigms, methods, models, species), not generic words.
CLOUD_TERMS = {
    # behaviours & paradigms
    "Grooming": r"groom", "Freezing": r"freezing", "Aggression": r"aggress",
    "Rearing": r"\brear(?:ing|s|ed)?\b", "Sniffing": r"sniff", "Mating": r"\bmating\b|copulat|mounting behav|sexual behav",
    "Digging": r"digging", "Climbing": r"climbing", "Immobility": r"immobil", "Locomotion": r"locomot",
    "Social interaction": r"social interaction", "Avoidance": r"avoidance",
    "Head-twitch": r"head.?twitch|\bhtr\b", "Scratching": r"scratching", "Nesting": r"nest(?:ing| building)",
    "Pup retrieval": r"pup retrieval", "Marble burying": r"marble bury", "Tail suspension": r"tail suspension",
    "Forced swim": r"forced swim", "Elevated plus maze": r"plus.?maze", "Open field": r"open.?field",
    "Three-chamber": r"three.?chamber", "Resident-intruder": r"resident.?intruder", "Novel object": r"novel object",
    "Fear conditioning": r"fear conditioning", "Place preference": r"place preference|\bcpp\b",
    "Rotarod": r"rotarod", "Gait": r"\bgait\b", "Startle": r"startle", "Feeding": r"feeding", "Licking": r"licking",
    # regions
    "Amygdala": r"amygdala", "Nucleus accumbens": r"nucleus accumbens|accumbens",
    "Prefrontal cortex": r"prefrontal|\bpfc\b", "Hippocampus": r"hippocamp",
    "BNST": r"\bbnst\b|bed nucleus of the stria", "Hypothalamus": r"hypothalam",
    "VTA": r"\bvta\b|ventral tegmental", "Striatum": r"striat", "Thalamus": r"\bthalam", "Insula": r"insula",
    "Cingulate": r"cingulate", "Periaqueductal gray": r"periaqueductal|\bpag\b", "Habenula": r"habenula",
    "Cerebellum": r"cerebell", "Dorsal raphe": r"dorsal raphe", "Locus coeruleus": r"locus coeruleus",
    # neuromodulators
    "Dopamine": r"dopamine", "Serotonin": r"serotonin|5-?ht\b", "Oxytocin": r"oxytocin", "GABA": r"\bgaba\b",
    "Glutamate": r"glutamate", "Opioid": r"opioid", "Cannabinoid": r"cannabinoid", "Corticosterone": r"corticosterone",
    "CRF": r"\bcrf\b|corticotropin", "Norepinephrine": r"norepinephrine|noradrenaline", "Acetylcholine": r"acetylcholine|cholinergic",
    # drugs / compounds
    "Psilocybin": r"psilocy", "Ketamine": r"ketamine", "Fentanyl": r"fentanyl", "Oxycodone": r"oxycodone",
    "Cocaine": r"cocaine", "Methamphetamine": r"methamphetamine", "Morphine": r"morphine",
    "Ethanol / alcohol": r"ethanol|alcohol", "Nicotine": r"nicotine", "DMT": r"\bdmt\b", "MDMA": r"\bmdma\b",
    "LSD": r"\blsd\b", "Diazepam": r"diazepam", "THC / cannabis": r"\bthc\b|cannabis|tetrahydrocannab",
    "Amphetamine": r"\bamphetamine",
    # methods / tools
    "DeepLabCut": r"deeplabcut|\bdlc\b", "SLEAP": r"\bsleap\b", "Optogenetics": r"optogenetic",
    "Fiber photometry": r"photometry", "DREADD": r"dreadd|chemogenetic", "Calcium imaging": r"calcium imaging|miniscope",
    "Two-photon": r"two.?photon|2.?photon", "Electrophysiology": r"electrophysiolog|patch.?clamp",
    "RNA-seq": r"rna.?seq|transcriptom", "SHAP": r"\bshap\b|shapley", "Random forest": r"random forest",
    "BORIS": r"\bboris\b", "EthoVision": r"ethovision", "Immunohistochemistry": r"immunohistochem|\bihc\b",
    "Machine learning": r"machine learning", "Pose estimation": r"pose estimation",
    # models / disease
    "Parkinson's": r"parkinson", "Alzheimer's": r"alzheimer", "Autism": r"autism|\basd\b",
    "Epilepsy": r"epilep|seizure", "Addiction": r"addiction", "Depression": r"depress", "Anxiety": r"anxiet",
    "Chronic stress": r"chronic stress", "Social defeat": r"social defeat", "Neuropathic pain": r"neuropathic pain",
    "Fragile X": r"fragile x", "TBI": r"\btbi\b|traumatic brain",
    # species / strains
    "Zebrafish": r"zebrafish", "C57BL/6": r"c57bl|\bc57\b", "CD1": r"\bcd.?1\b", "Sprague-Dawley": r"sprague.?dawley",
    "Gerbil": r"gerbil", "Prairie vole": r"prairie vole|\bvole", "Drosophila": r"drosophila", "Primate": r"macaque|primate",
    "Wistar rat": r"wistar", "Long-Evans": r"long.?evans", "BALB/c": r"\bbalb", "Transgenic": r"transgenic",
    "Knockout": r"knockout", "Crayfish / crustacean": r"crayfish|crustacean|\bcrab", "Songbird": r"songbird|zebra finch",
    # more behaviours & paradigms
    "Circling": r"circling", "Jumping / escape": r"jumping|escape behav", "Burrowing": r"burrow",
    "Ultrasonic vocalisation": r"ultrasonic vocal|\busv\b|\busvs\b", "Prepulse inhibition": r"prepulse|\bppi\b",
    "Social memory": r"social memory", "Social novelty": r"social novelty",
    "Object recognition": r"object recognition", "Spatial memory": r"spatial memory|spatial learning",
    "Water maze": r"water maze|morris water", "Y-maze": r"y.?maze", "T-maze": r"\bt.?maze", "Barnes maze": r"barnes maze",
    "Light-dark box": r"light.?dark box", "Operant task": r"operant|lever press|nose.?poke",
    "Self-administration": r"self.?administrat", "Reinstatement": r"reinstatement", "Extinction": r"extinction",
    "Wheel running": r"wheel running", "Stereotypy": r"stereotyp", "Catalepsy": r"cataleps", "Tremor": r"tremor",
    "Grimace scoring": r"grimace",
    # more regions
    "Orbitofrontal cortex": r"orbitofrontal|\bofc\b", "Lateral septum": r"lateral septum",
    "Preoptic area": r"preoptic|\bmpoa\b", "Paraventricular nucleus": r"paraventricular|\bpvn\b",
    "Substantia nigra": r"substantia nigra", "Entorhinal cortex": r"entorhinal", "Dentate gyrus": r"dentate gyrus",
    "Prelimbic / infralimbic": r"prelimbic|infralimbic", "Basolateral amygdala": r"basolateral",
    # more markers / neuromodulators
    "Estrogen": r"estrogen|estradiol", "Testosterone": r"testosterone", "Vasopressin": r"vasopressin",
    "BDNF": r"\bbdnf\b", "c-Fos": r"c.?fos\b", "Orexin": r"orexin|hypocretin",
    "Somatostatin": r"somatostatin|\bsst\b", "Parvalbumin": r"parvalbumin", "Cortisol": r"cortisol",
    "Endocannabinoid": r"endocannabinoid",
    # more drugs / compounds
    "Fluoxetine / SSRI": r"fluoxetine|\bssri\b", "Haloperidol": r"haloperidol", "Clozapine": r"clozapine",
    "MK-801": r"mk.?801|dizocilpine", "Scopolamine": r"scopolamine", "Caffeine": r"caffeine",
    "Buprenorphine": r"buprenorphine", "Naloxone / naltrexone": r"naloxone|naltrexone",
    # more methods / tools
    "UMAP": r"\bumap\b", "t-SNE": r"t.?sne\b", "HDBSCAN": r"hdbscan", "XGBoost": r"xgboost|gradient boost",
    "CNN / ResNet": r"\bcnn\b|resnet|convolutional neural", "Transformer": r"\btransformer", "Keypoint tracking": r"keypoint|key.?point",
    "Bounding box": r"bounding box|\bbbox\b", "MoSeq": r"moseq", "VAME": r"\bvame\b", "B-SOiD": r"b.?soid",
    "A-SOiD": r"a.?soid", "DANNCE": r"dannce", "Lightning Pose": r"lightning ?pose", "AnyMaze": r"any.?maze",
    "TopScan / CleverSys": r"topscan|clever ?sys", "ezTrack": r"eztrack", "Unsupervised": r"unsupervised",
    "SVM": r"\bsvm\b|support vector",
    # more models / disease
    "Schizophrenia": r"schizophren", "PTSD": r"\bptsd\b|post.?traumatic", "ADHD": r"\badhd\b",
    "Huntington's": r"huntington", "Stroke / ischemia": r"\bstroke\b|ischemi|ischaemi", "Rett syndrome": r"rett syndrome|\bmecp2\b",
    "Maternal separation": r"maternal separation|early.?life advers|early life stress", "Neuroinflammation": r"neuroinflamm",
    "Obesity / diet": r"obesity|diet.?induced|high.?fat", "Aging": r"\baging\b|\baged\b|ageing", "Sleep / circadian": r"\bsleep\b|circadian",
}


def full_text(fp):
    try:
        r = subprocess.run(["pdftotext", fp, "-"], capture_output=True, timeout=90)
        return r.stdout.decode("utf-8", "ignore").lower()
    except Exception:
        return ""


def df_counts(texts, lex):
    c = collections.Counter()
    for t in texts:
        for label, pat in lex.items():
            if re.search(pat, t):
                c[label] += 1
    return [[k, v] for k, v in sorted(c.items(), key=lambda kv: -kv[1]) if v]


def main():
    if not os.path.isdir(PDF_DIR):
        print(f"[corpus] paper dir not found ({PDF_DIR}); keeping existing {OUT}."); return
    pdfs = [os.path.join(PDF_DIR, f) for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
    if not pdfs:
        print(f"[corpus] no PDFs in {PDF_DIR}; keeping existing {OUT}."); return

    texts = [full_text(fp) for fp in pdfs]
    cloud = collections.Counter()      # document frequency over the curated concept gazetteer
    for t in texts:
        for term, pat in CLOUD_TERMS.items():
            if re.search(pat, t):
                cloud[term] += 1

    data = {
        "generated": date.today().strftime("%B %d, %Y"),
        "n_pdfs": len(pdfs),
        "behaviors": df_counts(texts, BEHAVIORS),
        "regions": df_counts(texts, REGIONS),
        "methods": df_counts(texts, METHODS),
        "diseases": df_counts(texts, DISEASES),
        "wordcloud": [[term, n] for term, n in cloud.most_common(200) if n >= 5],
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    print(f"[corpus] wrote {OUT}: {len(pdfs)} PDFs | "
          f"behaviors {len(data['behaviors'])} | regions {len(data['regions'])} | "
          f"methods {len(data['methods'])} | diseases {len(data['diseases'])} | "
          f"cloud {len(data['wordcloud'])} words")


if __name__ == "__main__":
    main()
