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
import os, re, sys, json, hashlib, subprocess, collections, itertools
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
    # miniscope / two-photon fold in here rather than getting their own bars, so a
    # calcium-imaging paper is not counted three times across the panel.
    "Calcium imaging": r"calcium imaging|miniscope|two.?photon|2.?photon",
    "Electrophysiology": r"electrophysiolog|patch.?clamp|single.?unit|in vivo record",
    "Transcriptomics": r"rna.?seq|transcriptom|single.?cell rna",
    "SHAP explainability": r"\bshap\b|shapley",
    "BORIS annotation": r"\bboris\b",
    "EthoVision": r"ethovision",
    "Viral tracing (AAV)": r"\baav\b|adeno.?associated|retrograde trac|anterograde trac",
    "Immunohistochemistry": r"immunohistochem|\bihc\b",
    "c-Fos mapping": r"c.?fos\b",
    "EEG / LFP recording": r"\beeg\b|local field potential|\blfp\b",
    "Ultrasonic vocalisation": r"ultrasonic vocal|\busvs?\b",
    "AnyMaze": r"any.?maze",
    "B-SOiD": r"b.?soid",
    "Keypoint-MoSeq": r"moseq",
    "Microdialysis": r"microdialysis",
    "qPCR": r"\bqpcr\b|quantitative pcr|rt.?pcr",
    "A-SOiD": r"a.?soid",
    "VAME": r"\bvame\b",
    "Western blot": r"western blot",
    "UMAP / t-SNE": r"\bumap\b|t.?sne\b",
    "TopScan / CleverSys": r"topscan|clever ?sys",
    "Telemetry": r"telemetr",
    "XGBoost": r"xgboost|gradient boost",
    "Lightning Pose": r"lightning ?pose",
    "DANNCE": r"dannce",
    "ezTrack": r"eztrack",
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
    "Schizophrenia": r"schizophren",
    "Fear conditioning": r"fear conditioning",
    # "aged 8 weeks" is husbandry, not ageing research: \baged\b matched 45 papers
    # that way, so require the research sense.
    "Ageing": r"\baging\b|\bageing\b|age.?related",
    "Sleep / circadian": r"sleep deprivation|circadian rhythm|sleep.?wake",
    "Early-life stress": r"maternal separation|early.?life (?:stress|advers)|limited bedding",
    "Social defeat": r"social defeat",
    "PTSD": r"\bptsd\b|post.?traumatic stress",
    "Neuroinflammation": r"neuroinflamm",
    "Obesity / diet": r"obesity|diet.?induced|high.?fat diet",
    "Stroke / ischemia": r"\bstroke\b|ischemi|ischaemi",
    "Prenatal exposure": r"prenatal|in utero|gestational exposure",
    "Neuropathic pain": r"neuropathic pain",
    "Huntington's": r"huntington",
    "ADHD": r"\badhd\b",
    "Traumatic brain injury": r"\btbi\b|traumatic brain injur",
    "Rett syndrome": r"rett syndrome|\bmecp2\b",
    "Fragile X": r"fragile x|\bfmr1\b",
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
    "Water maze": r"water maze|morris water", "Y-maze": r"\by.?maze", "T-maze": r"\bt.?maze", "Barnes maze": r"barnes maze",
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


# --- "behaviours automated": what SimBA was actually used to score -----------------
# Whole-document keyword counting cannot answer this -- "freezing" appears in papers
# that never built a freezing classifier. So a behaviour is credited to a study only
# when its term sits BOTH (a) inside a text window around a SimBA mention and
# (b) within NEAR characters of a scoring/classifier cue. (a) alone credits husbandry
# and unrelated assays ("nicotine in the drinking water"); (b) alone credits any ML
# mentioned anywhere in the paper.
CTX_BEFORE, CTX_AFTER, NEAR = 400, 700, 160

# A SimBA mention inside the reference list is the Goodwin/Nilsson citation itself;
# its neighbours are unrelated references, so those windows are dropped.
# NB the 2020 preprint's own title contains "complex social behaviors in experimental
# animals" -- without it here, every paper that merely cites SimBA was credited with
# automating social behaviour.
SIMBA_CITE = re.compile(
    r"as a platform for explainable|simple behavioral analysis \(simba\) as a"
    r"|goodwin,? ?n\.?\s?l|nilsson,? ?s\.?\s?r\.?\s?o|simba: a novel"
    r"|open.?source deep learning based framework"
    r"|open source toolkit for computer classification"
    r"|toolkit for computer classification of complex social")
# Cues that mark text as "this is a behaviour being read out". Three groups, because
# studies phrase it three ways: trained classifiers ("scored", "classifier for"),
# ethogram definitions ("'attack' was defined as"), and kinematic/ROI readouts
# ("time spent", "total distance was measured") -- the last two carry no ML verb at
# all, and requiring one silently dropped every ROI-only and ethogram-table study.
SCORE_CUE = re.compile(
    r"classif|scor(?:e|ed|es|ing)\b|annotat|detect|quantif|\btrain(?:ed|ing)\b"
    r"|predict|label(?:l?ed|l?ing)\b|ethogram|random forest|\bbouts?\b"
    r"|behaviou?rs? (?:such as|of interest|included|were)|automated|automatic"
    r"|defined as|was defined|were defined|definition of"
    r"|measur|calculat|comput(?:e|ed|ing)\b|time spent|duration (?:of|and)|frequency of"
    r"|analy[sz]ed (?:in|with|using)|readouts?\b")

# A behaviour named inside a bibliography entry is someone else's study, not this one.
# SIMBA_CITE only drops the SimBA citation itself; a window can still reach into the
# neighbouring references, where "(2020). ... duration of attack ... eNeuro 7(5)" reads
# as a behaviour. Only near-unambiguous reference markers: a DOI or a volume(issue)
# citation. NOT "(2020)." -- that is also how ordinary methods prose ends a sentence.
REF_ZONE = re.compile(r"doi\.org/|\bdoi:|\b\d{4};\s?\d+\(|\b\d+\(\d+\),\s?\d+")

# label -> (surface forms, family). Families group the ethogram for the docs caption.
BEHAV_AUTOMATED = {
 "Social interaction / approach":
   (r"social (?:interaction|investigat|approach|contact|behavio|preference|proximity)|allogroom|social novelty|crawling", "Social"),
 "Anogenital / body sniffing":
   (r"sniff|anogenital|ano.?genital|nose.?to.?nose|head.?to.?head|face.?to.?face|nosing", "Social"),
 # "following" is a preposition ("following model training") and a list introducer
 # ("the following behaviors:") far more often than it is the behaviour, so it only
 # counts with an explicit object or inside a classifier list.
 "Following / chasing / pursuit":
   (r"\bchasing\b|\bchase\b|pursuit"
    r"|(?<!the )(?<!these )(?<!as )following (?:and (?:circling|sniffing|chas)|classifiers?\b)"
    r"|follow(?:ing|ed) (?:the )?(?:conspecific|intruder|stimulus|demonstrator|partner|another|other mouse)", "Social"),
 "Mating / mounting":
   (r"\bmounting\b(?! (?:magnet|camera|the))|copulat|intromission|mating behavio|sexual behavio|\bthrust", "Social"),
 "Pup retrieval / maternal care":
   (r"pup retrieval|retriev\w* (?:the )?pups?|maternal (?:behavio|care)|nest building|nest attendance|dam.?pup|carrying|maternal approach|nest shift", "Social"),
 "Attack / fighting / biting":
   (r"\battack|fighting|\bbit(?:e|es|ing)\b|aggressive behavio|\bstrik(?:e|es|ing)\b|lateral threat"
    r"|\btussl|\blung(?:e|es|ing)\b|offensive|\bpinning\b|\bgrappling\b", "Aggression"),
 "Tail rattling / dominance display":
   (r"tail rattl|\bdominance\b|dominant (?:male|mice|mouse|animal)|submissi|threat display|\bboxing\b", "Aggression"),
 "Freezing": (r"freezing(?! (?:microtome|point))|\bfreeze\b", "Fear / defence"),
 # Darting is its own explicitly defined fear response ("movement across the chamber
 # at or exceeding 20 cm/s"), not a flavour of escape.
 "Darting": (r"\bdarting\b|\bdarts?\b(?= (?:were|was|behavio|bout))", "Fear / defence"),
 "Avoidance / escape / flight":
   (r"avoidance|\bescape|\bflee|fleeing|\brunaway\b|\bflight\b|\bretreat", "Fear / defence"),
 "Defensive posture / risk assessment":
   (r"defensive (?:behavio|postur|attack|burying)|\bupright\b|risk assessment"
    r"|stretch.?attend|head dip|\bdipping\b|\bcrouch", "Fear / defence"),
 # No leading \b: the corpus writes "selfgrooming" as one word. "allogrooming" is
 # social grooming and is counted under social interaction instead.
 "Grooming": (r"(?<!allo)(?<!allo-)groom", "Self-directed"),
 "Rearing": (r"\brear(?:ing|s|ed)?\b(?! (?:environment|paw|left|right|limb))", "Self-directed"),
 "Head-twitch response": (r"head.?twitch|\bhtr\b", "Self-directed"),
 "Digging / burrowing": (r"\bdigging\b|burrow|marble bury|\bburying\b", "Self-directed"),
 "Stereotypy (circling / pacing / Straub tail)":
   (r"\bcircling\b|straub|\btremor|catalep|uncoordinated walking|\bpacing\b|\bswaying\b", "Self-directed"),
 "Hind-limb clasping": (r"clasping", "Self-directed"),
 "Locomotion / distance travelled":
   (r"locomot|distance (?:travel|moved|travell)|total distance|movement distance|ambulat", "Locomotion / motor"),
 "Immobility / motionless":
   # "resting" is a scored state in the fish/rat ethograms, but "resting state"
   # is LFP/fMRI and "initial resting state" is the head-twitch baseline.
   (r"immobility|\bimmobile\b|motionless|\bresting\b(?! state)", "Locomotion / motor"),
 "Gait / balance beam":
   # "stride" alone also matched a convolution stride in a deep-learning methods section.
   # "walking behaviour" is scored directly; bare "walking" also matches a crab's
   # "walking appendages", so keep the noun.
   (r"\bgait\b|balance beam|beam walking|rotarod|walking (?:pattern|behavio|time)"
    r"|footfall|stride length|foot.?slip|pole test|wire hang", "Locomotion / motor"),
 "Climbing / jumping": (r"climbing|\bjump(?:ing|s)?\b", "Locomotion / motor"),
 "Wheel running": (r"wheel running", "Locomotion / motor"),
 # Aquatic species: swimming style, station-holding against flow, hovering in place.
 # Bare "swimming" is mostly the species ("swimming crab Portunus") or anatomy
 # ("swimming limbs"), so require a readout noun after it.
 "Swimming / rheotaxis / hovering":
   (r"swimming (?:behavio|time|activity|pattern|style|bout)|parallel swim"
    r"|\brheotaxis\b|station.?holding|\bhovering\b|bottom.?dwelling", "Locomotion / motor"),
 "Feeding / licking / drinking":
   (r"\blick(?:ing|s)?\b|feeding behavio|food intake|\beating\b|drinking behavio|appetitive|ingestion|\bnursing\b", "Feeding / reward"),
 "Foraging / food handling":
   (r"\bgnaw|\btearing\b|foraging|food handling|\bpecking\b|prey capture|\bhunting\b|predatory", "Feeding / reward"),
 "Operant / self-administration":
   (r"self.?administrat|operant|lever.?press|nose.?poke|place preference|reinstatement|drug.?seeking|drug.?taking", "Feeding / reward"),
 # Cleaner-fish mutualism: client "jolts" index cheating by the cleaner.
 "Cleaning interaction / jolts (fish)":
   (r"\bjolts?\b|cleaning (?:behavio|interaction)|client interaction", "Feeding / reward"),
 "Object exploration / novel object":
   (r"object (?:exploration|interaction|investigat|contact|approach)|object recognition(?! benchmark)|novel object|exploratory behavio", "Exploration"),
 "Zone / ROI occupancy":
   (r"region.?of.?interest|\brois?\b|time.?in.?zone|zone (?:occupancy|entr|time)|time in (?:the )?(?:cent(?:er|re)|corner)|\bcrossings?\b", "Exploration"),
 # Anchor the maze names: unanchored "y.?maze" also matches "anymaze", the tracking software.
 "Whisking / head scanning":
   (r"whisking|head scanning", "Exploration"),
 "Maze arm entries / spatial task":
   (r"arm entr|\by.?maze|\bt.?maze|water maze|barnes maze|plus.?maze|spatial (?:memory|learning) task", "Exploration"),
}


def flow(t):
    """pdftotext output -> single-line text: de-hyphenate line breaks, collapse space."""
    return re.sub(r"\s+", " ", re.sub(r"-\s*\n\s*", "", t.replace("­", "")))


def simba_context(t):
    """Concatenated windows around every SimBA mention that is not a bibliography entry.
    Edges snap outward to whitespace: a window cutting "classi|fications" in half hides
    the cue that the behaviour term next to it depends on."""
    wins = []
    for m in re.finditer(r"simba|simple behaviou?ral analysis", t):
        s = m.start()
        if SIMBA_CITE.search(t[max(0, s - 170):s + 170]):
            continue
        a, b = max(0, s - CTX_BEFORE), min(len(t), s + CTX_AFTER)
        a = t.rfind(" ", 0, a) + 1 if a else 0
        b = t.find(" ", b)
        wins.append(t[a:b if b != -1 else len(t)])
    return " | ".join(wins)


def study_clusters(texts):
    """Group files that are the same study -- a preprint and its published version, or
    the same PDF saved twice -- so document frequency counts each study once.
    Containment of 9-word shingles; step 1 so a 1-2 word offset cannot de-align them."""
    def sh(t, k=9, cap=6000):
        w = re.findall(r"[a-z]+", t)[:cap]
        return {" ".join(w[i:i + k]) for i in range(max(0, len(w) - k))}

    S = [sh(t) for t in texts]
    par = list(range(len(texts)))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x

    for a, b in itertools.combinations(range(len(texts)), 2):
        if S[a] and S[b] and len(S[a] & S[b]) / min(len(S[a]), len(S[b])) > 0.25:
            ra, rb = find(a), find(b)
            if ra != rb:
                par[ra] = rb
    g = collections.defaultdict(list)
    for i in range(len(texts)):
        g[find(i)].append(i)
    return list(g.values())


def behaviours_automated(texts):
    """-> ([[label, n_studies, family], ...] desc, n_studies_with_simba_context,
    {label: [paper_index, ...]}). The index is the study's representative paper."""
    flowed = [flow(t) for t in texts]
    counts, ex, n = collections.Counter(), collections.defaultdict(list), 0
    for group in study_clusters(flowed):
        blob = " | ".join(filter(None, (simba_context(flowed[i]) for i in group)))
        if not blob:
            continue
        n += 1
        for label, (pat, _) in BEHAV_AUTOMATED.items():
            for m in re.finditer(pat, blob):
                near = blob[max(0, m.start() - NEAR):m.start() + NEAR]
                if SCORE_CUE.search(near) and not REF_ZONE.search(near):
                    counts[label] += 1
                    ex[label].append(group[0])
                    break
    rows = [[lab, c, BEHAV_AUTOMATED[lab][1]] for lab, c in counts.most_common()]
    return rows, n, {lab: pick_examples(lab, ex[lab]) for lab, _, _ in rows}


def full_text(fp):
    try:
        r = subprocess.run(["pdftotext", fp, "-"], capture_output=True, timeout=90)
        return r.stdout.decode("utf-8", "ignore").lower()
    except Exception:
        return ""


# Enough of each paper's opening to contain its title once journal furniture
# ("contents lists available at sciencedirect ...") is allowed for. The renderer
# matches these against the sheet's TITLE column to name papers in tooltips.
HEAD_CHARS = 700
EXAMPLE_CAP = 4          # papers listed per label in a tooltip


def paper_head(t):
    """Lowercased opening with punctuation collapsed to spaces. Spaces are kept so the
    renderer can both substring-match a title and fall back to word overlap."""
    return re.sub(r"[^a-z0-9]+", " ", flow(t)[:3000].lower()).strip()[:HEAD_CHARS]


def pick_examples(label, idxs):
    """A deterministic per-label sample of the matching papers.

    Taking the first few by index meant the same paper headlined nearly every label
    (paper 0 mentions most things). Ordering by a hash of label+index decorrelates the
    choice across labels while staying stable across rebuilds."""
    return sorted(idxs, key=lambda i: hashlib.md5(f"{label}:{i}".encode()).hexdigest())[:EXAMPLE_CAP]


def df_counts(texts, lex):
    """-> ([[label, count], ...] desc, {label: [paper_index, ...]}) -- the indices are
    a sample of the papers matching each label, for tooltip examples."""
    c = collections.Counter()
    hits = collections.defaultdict(list)
    for i, t in enumerate(texts):
        for label, pat in lex.items():
            if re.search(pat, t):
                c[label] += 1
                hits[label].append(i)
    rows = [[k, v] for k, v in sorted(c.items(), key=lambda kv: -kv[1]) if v]
    return rows, {k: pick_examples(k, hits[k]) for k, _ in rows}


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

    behav_rows, n_behav_studies, behav_ex = behaviours_automated(texts)
    beh, beh_ex = df_counts(texts, BEHAVIORS)
    reg, reg_ex = df_counts(texts, REGIONS)
    met, met_ex = df_counts(texts, METHODS)
    dis, dis_ex = df_counts(texts, DISEASES)

    data = {
        "generated": date.today().strftime("%B %d, %Y"),
        "n_pdfs": len(pdfs),
        "behaviours_automated": behav_rows,
        "n_behaviour_studies": n_behav_studies,
        "behaviors": beh,
        "regions": reg,
        "methods": met,
        "diseases": dis,
        # Normalised paper openings, and which papers back each label. The renderer
        # resolves these to curated titles via the sheet; kept out of the label rows
        # so their shape stays [label, count(, family)].
        "papers": [paper_head(t) for t in texts],
        "examples": {"behaviours_automated": behav_ex, "behaviors": beh_ex,
                     "regions": reg_ex, "methods": met_ex, "diseases": dis_ex},
        "wordcloud": [[term, n] for term, n in cloud.most_common(200) if n >= 5],
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    print(f"[corpus] wrote {OUT}: {len(pdfs)} PDFs | "
          f"behaviours automated {len(behav_rows)} labels over {n_behav_studies} studies | "
          f"behaviors {len(data['behaviors'])} | regions {len(data['regions'])} | "
          f"methods {len(data['methods'])} | diseases {len(data['diseases'])} | "
          f"cloud {len(data['wordcloud'])} words")


if __name__ == "__main__":
    main()
