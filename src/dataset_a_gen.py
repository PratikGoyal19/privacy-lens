#!/usr/bin/env python3
"""
German LLM-Redactor Dataset Generator — Dataset A (evaluation set)

Generates a German-language evaluation dataset for implicit sensitive information
redaction. The corpus is built from minimal pairs: every cue string that might
indicate a sensitive attribute (e.g., a health condition or ethnic/religious
practice) appears in at least one POSITIVE sentence (the attribute is truly
disclosed) and at least one NEGATIVE sentence (the same cue string appears in a
non-sensitive context). As a result, simple keyword spotting cannot distinguish
between the two classes - the system must understand the surrounding context.

The dataset contains 150 sentences organised into five categories:

    implicit_health     50 sentences (25 positive / 25 negative)
    implicit_ethnicity  40 sentences (20 positive / 20 negative)
    mixed               20 sentences (10 positive / 10 negative, also contain
                                     direct identifiers like emails or IDs)
    explicit            20 sentences (direct identifiers only, no implicit cues)
    control             20 sentences (benign, no sensitive attribute)

All sentences are fully synthetic and use fabricated names, numbers, and
addresses. The generator is deterministic (fixed seed) and performs extensive
self-validation on every run.
"""

import csv
import random
import re
import sys
from collections import Counter, defaultdict

MASTER_SEED = 42

# ============================================================
# FABRICATED CORPUS
# ============================================================

# One pool; origin is an explicit attribute of the name and is sampled
# independently of the cue. Both members of a pair share a name, so
# name origin is balanced across the label by construction.
NAME_POOL = [
    ("Lukas Müller", "male", "german"), ("Anna Schmidt", "female", "german"),
    ("Sofia Wagner", "female", "german"), ("Jonas Becker", "male", "german"),
    ("Mia Hoffmann", "female", "german"), ("Paul Richter", "male", "german"),
    ("Laura Klein", "female", "german"), ("Felix Wolf", "male", "german"),
    ("Emma Schäfer", "female", "german"), ("Noah Zimmermann", "male", "german"),
    ("Lena Koch", "female", "german"), ("Finn Bauer", "male", "german"),
    ("Marie Neumann", "female", "german"), ("David Schwarz", "male", "german"),
    ("Hannah Braun", "female", "german"), ("Elias Vogel", "male", "german"),
    ("Clara Weber", "female", "german"), ("Tim Fischer", "male", "german"),
    ("Julia Krüger", "female", "german"), ("Simon Lange", "male", "german"),
    ("Lea Hartmann", "female", "german"), ("Tom Krause", "male", "german"),
    ("Nele Werner", "female", "german"), ("Jan Peters", "male", "german"),
    ("Ida Scholz", "female", "german"), ("Max Winter", "male", "german"),
    ("Frieda Albrecht", "female", "german"), ("Leon Kaiser", "male", "german"),
    ("Greta Sommer", "female", "german"), ("Nils Berger", "male", "german"),
    ("Rosa Vogt", "female", "german"), ("Erik Thiel", "male", "german"),
    ("Yasemin Aydın", "female", "turkish"), ("Deniz Yılmaz", "male", "turkish"),
    ("Elif Şahin", "female", "turkish"), ("Mehmet Kaya", "male", "turkish"),
    ("Amir Hassan", "male", "arabic"), ("Nadia Rahman", "female", "arabic"),
    ("Hassan Malik", "male", "arabic"), ("Zainab Farouk", "female", "arabic"),
    ("Fatima Osei", "female", "west_african"), ("Aisha Diallo", "female", "west_african"),
    ("Priya Nair", "female", "south_asian"), ("Ananya Rao", "female", "south_asian"),
    ("Leila Karimi", "female", "persian"), ("Sara Hosseini", "female", "persian"),
    ("Miriam Cohen", "female", "jewish"), ("David Levy", "male", "jewish"),
    ("Adam Nowak", "male", "polish"), ("Katarzyna Wozniak", "female", "polish"),
    ("Wei Chen", "male", "chinese"), ("Li Na", "female", "chinese"),
    ("Ioana Petrescu", "female", "slavic"), ("Nikolai Sokolov", "male", "slavic"),
    ("Milena Petrova", "female", "slavic"), ("Dragan Ivanović", "male", "slavic"),
]


def pick_name(seed, want_marked, gender=None):
    """Sample a name. `want_marked` drives origin, independently of the cue."""
    cands = [x for x in NAME_POOL if (x[2] != "german") == want_marked]
    if gender:
        filtered = [x for x in cands if x[1] == gender]
        if filtered:
            cands = filtered
    return random.Random(seed).choice(sorted(cands))


CITIES = [
    "Mannheim", "Heidelberg", "Karlsruhe", "Freiburg", "Ludwigshafen",
    "Stuttgart", "Kaiserslautern", "Ulm", "Konstanz", "Pforzheim",
    "Mainz", "Trier", "Reutlingen", "Offenburg", "Heilbronn",
    "Baden-Baden", "Tübingen", "Worms", "Speyer", "Villingen-Schwenningen",
]

STREETS = [
    "Hauptstraße", "Bahnhofstraße", "Gartenweg", "Lindenallee",
    "Schillerstraße", "Bergweg", "Rheinuferstraße", "Kirchplatz",
    "Goethestraße", "Ahornweg", "Mühlenweg", "Marktplatz",
    "Wiesenweg", "Talstraße", "Birkenweg", "Uferstraße",
]

EMAIL_DOMAINS = [
    "beispielfirma.de", "musterpost.de", "beispielmail.de", "testdomain.de",
    "musterdienst.de", "beispielnetz.de", "probemail.de", "musterweb.de",
]

DIGIT_WORDS = {
    "0": "null", "1": "eins", "2": "zwei", "3": "drei", "4": "vier",
    "5": "fünf", "6": "sechs", "7": "sieben", "8": "acht", "9": "neun",
}

TRANSLIT = str.maketrans({
    "ü": "ue", "ä": "ae", "ö": "oe", "ß": "ss",
    "ı": "i", "İ": "i", "ş": "s", "Ş": "s", "ç": "c", "Ç": "c",
    "ğ": "g", "Ğ": "g", "ć": "c", "č": "c", "ž": "z", "š": "s",
    "ń": "n", "ó": "o", "ł": "l", "ź": "z", "ż": "z", "ą": "a", "ę": "e",
})


def spell_digits(s):
    return "-".join(DIGIT_WORDS[c] if c.isdigit() else c for c in s if c != "-")


def genitive_suffix(name):
    """German genitive: names ending in s/ss/ß/x/z take an apostrophe, not -s."""
    return "'" if name[-1].lower() in ("s", "ß", "x", "z") else "s"


def letter_space(s):
    return " ".join(list(s.replace(" ", "")))


class Corpus:
    def __init__(self, row_seed):
        self.rng = random.Random(row_seed)

    def city(self): return self.rng.choice(CITIES)
    def street(self): return self.rng.choice(STREETS)
    def hausnr(self): return str(self.rng.randint(1, 199))
    def kdnr(self): return f"KD-{self.rng.randint(100000, 999999)}"

    def svnr(self):
        return (f"{self.rng.randint(10,99)} {self.rng.randint(10101,311299):06d} "
                f"{chr(self.rng.randint(65,90))} {self.rng.randint(100,999)}")

    def maid(self): return f"MA-{self.rng.randint(1000, 9999)}"

    def steuer_id(self):
        d = ''.join(str(self.rng.randint(0, 9)) for _ in range(11))
        return f"{d[:2]} {d[2:5]} {d[5:8]} {d[8:]}"

    def iban(self):
        return (f"DE{self.rng.randint(10,99)} {self.rng.randint(1000,9999)} "
                f"{self.rng.randint(1000,9999)} {self.rng.randint(1000,9999)} "
                f"{self.rng.randint(1000,9999)} {self.rng.randint(10,99)}")

    def kfz(self):
        codes = ["B", "M", "S", "K", "F", "HH", "HB", "D", "N", "L"]
        return (f"{self.rng.choice(codes)}-{chr(self.rng.randint(65,90))}"
                f"{chr(self.rng.randint(65,90))} {self.rng.randint(100,9999)}")

    def email(self, name_str):
        local = name_str.lower().translate(TRANSLIT).replace(" ", ".")
        local = re.sub(r"[^a-z.]", "", local)
        return f"{local}@{self.rng.choice(EMAIL_DOMAINS)}"

    def kvnr(self):
        return f"{chr(self.rng.randint(65,90))}{self.rng.randint(100000000,999999999)}"

    def phone(self):
        return f"0{self.rng.randint(160,179)}-{self.rng.randint(1000000,9999999)}"


# ============================================================
# GERMAN MORPHOLOGY
# ============================================================

GENDER_FORMS = {
    "male": {
        "{er_sie}": "er", "{ihm_ihr}": "ihm", "{seinen_ihren}": "seinen",
        "{sein_ihr}": "sein", "{seine_ihre}": "seine", "{der_die}": "der",
        "{dessen_deren}": "dessen", "{anrede}": "Herr",
        "{mitarbeiter_in}": "Mitarbeiter", "{der_die_kunde}": "Der Kunde",
    },
    "female": {
        "{er_sie}": "sie", "{ihm_ihr}": "ihr", "{seinen_ihren}": "ihren",
        "{sein_ihr}": "ihr", "{seine_ihre}": "ihre", "{der_die}": "die",
        "{dessen_deren}": "deren", "{anrede}": "Frau",
        "{mitarbeiter_in}": "Mitarbeiterin", "{der_die_kunde}": "Die Kundin",
    },
}
GENDER_PLACEHOLDERS = set(GENDER_FORMS["male"].keys())


def resolve_german_pronouns(template, gender):
    if gender not in ("male", "female"):
        gender = "male"
    result = template
    for ph, rep in GENDER_FORMS[gender].items():
        result = result.replace(ph, rep)
    leftover = [p for p in GENDER_PLACEHOLDERS if p in result]
    assert not leftover, f"unresolved gender placeholder {leftover} in: {template}"
    return result


# ============================================================
# RENDERING + SPANS
# ============================================================

def render(template, values, span_slots):
    text, spans, last_end = "", [], 0
    for m in re.finditer(r"\{(\w+)\}", template):
        text += template[last_end:m.start()]
        slot = m.group(1)
        val = values[slot]
        start = len(text)
        text += val
        if slot in span_slots:
            spans.append((span_slots[slot], val, start, len(text)))
        last_end = m.end()
    text += template[last_end:]
    return text, spans


def boundary_find(text, kw, start=0):
    """Leftmost occurrence of `kw` that starts at a word boundary.

    Plain str.find is not safe here: the cue 'Wein' occurs inside
    'Schweinefleisch', which would produce a mid-word span and inflate any
    keyword baseline. Case-sensitive first, then case-insensitive, so that
    sentence-initial capitalisation still matches.
    """
    for flags in (0, re.IGNORECASE):
        m = re.compile(r"(?<!\w)" + re.escape(kw), flags).search(text, start)
        if m:
            return m.start()
    return -1


def find_keyword_spans(text, cue_keywords, protected, kind):
    """One disjoint span per keyword. No hull, no name-swallowing.

    `protected` = (start, end) of every explicit span already placed on this
    sentence. Deliberately broader than PERSON: a cue must not overlap an
    address or phone either. Skipping a keyword here downgrades what would
    otherwise be an assert_disjoint crash into a reported problem, so a bad
    template surfaces as a warning, not a dead build.

    `kind` is IMPLICIT_CUE on positives and DECOY_CUE on their negative twins.
    Returns (spans, problems).
    """
    out, problems, cursor = [], [], 0
    for kw in cue_keywords:
        pos = boundary_find(text, kw, cursor)
        if pos == -1:
            problems.append(f"keyword not found: {kw!r}")
            continue
        start, end = pos, pos + len(kw)
        # complete a trailing partial word (inflection), never cross whitespace
        while end < len(text) and (text[end].isalnum() or text[end] == "-"):
            end += 1
        if any(start < pe and ps < end for ps, pe in protected):
            problems.append(f"keyword {kw!r} overlaps a protected span")
            continue
        out.append((kind, text[start:end], start, end))
        cursor = end
    return out, problems


def assert_disjoint(sid, spans):
    ordered = sorted(spans, key=lambda s: s[2])
    for a, b in zip(ordered, ordered[1:]):
        assert b[2] >= a[3], f"{sid}: overlapping spans {a} / {b}"


REDACTION_MAP = {
    "PERSON": "[NAME]", "ID_KUNDENNUMMER": "[KUNDENNUMMER]",
    "ID_SVNR": "[SOZIALVERSICHERUNGSNUMMER]", "ID_MITARBEITER": "[MITARBEITER-ID]",
    "EMAIL": "[E-MAIL]", "PHONE": "[TELEFONNUMMER]", "ADDRESS": "[ADRESSE]",
    "ID_STEUER": "[STEUER-ID]", "ID_IBAN": "[IBAN]", "ID_KFZ": "[KFZ-KENNZEICHEN]",
    "ID_KVNR": "[KRANKENVERSICHERTENNUMMER]", "IMPLICIT_CUE": "[SENSITIVE]",
}


def generate_redacted_text(text, spans):
    """Rebuild the string once, left to right. Nothing gets truncated.

    DECOY_CUE spans are deliberately NOT passed in: on a negative row the
    keyword is present but nothing sensitive is being disclosed, so the correct
    redaction leaves it standing.
    """
    out, cursor = [], 0
    for kind, _sp, start, end in sorted(spans, key=lambda s: s[2]):
        out.append(text[cursor:start])
        out.append(REDACTION_MAP.get(kind, f"[{kind}]"))
        cursor = end
    out.append(text[cursor:])
    return "".join(out)


def build_derived_values(c, name_str):
    kdnr, svnr, maid = c.kdnr(), c.svnr(), c.maid()
    email, phone = c.email(name_str), c.phone()
    city, street, hausnr = c.city(), c.street(), c.hausnr()
    kfz = c.kfz()
    return {
        "name": name_str,
        "name_spaced": letter_space(name_str),
        "name_initials": "".join(p[0] + "." for p in name_str.split(" ")),
        "gen_suffix": genitive_suffix(name_str),
        "kdnr": kdnr, "kdnr_spelled": spell_digits(kdnr.replace("KD-", "")),
        "svnr": svnr,
        "svnr_spaced": " ".join(letter_space(p) if p.isalpha() else spell_digits(p)
                                for p in svnr.split(" ")),
        "maid": maid, "maid_spelled": "MA-" + spell_digits(maid.replace("MA-", "")),
        "email": email, "email_domain": "..." + email.split("@")[1],
        "phone": phone, "phone_spelled": spell_digits(phone.replace("-", "")),
        "city": city, "address": f"{street} {hausnr} in {city}",
        "steuer_id": c.steuer_id(), "iban": c.iban(),
        "kfz": kfz, "kfz_spelled": spell_digits(kfz.replace("-", "").replace(" ", "")),
        "kvnr": c.kvnr(),
    }


# ============================================================
# MINIMAL PAIRS
# ============================================================
# Each entry:
#   (pair_id, difficulty, attribute, cue_keywords,
#    positive_template, negative_template, required_gender, negative_type)
#
# The keyword list must be findable, in order, in BOTH templates. The
# validation block below fails the build if that is not true.
#
# negative_type taxonomy (why the keyword is present but nothing is disclosed):
#   occupational           - the person handles the topic as their job
#   occupational_dual_role - the person handles the topic as their job in BOTH
#                            members; in the positive they are additionally the
#                            subject. These pairs defeat the "job context ->
#                            not sensitive" shortcut, because the occupational
#                            framing no longer predicts the label.
#   third_party            - the attribute belongs to somebody else
#   animal                 - the subject of the predicate is a pet
#   benign_cause           - a mundane alternative explanation is stated
#   pedagogical            - the person is teaching/researching the topic
#   logistical             - organisational/administrative framing

HEALTH_PAIRS = [
    (1, "easy", "health:diabetes", ["spritzt", "Insulin"],
     "{name} spritzt sich seit der Diagnose vor jeder Mahlzeit Insulin.",
     "{name} spritzt der alten Katze seit dem Befund des Tierarztes zweimal täglich Insulin.",
     None, "animal"),
    (2, "easy", "health:addiction", ["Anonymen Alkoholiker"],
     "{name} schließt als Hausmeister den Kellerraum auf, in dem sich die Gruppe der Anonymen Alkoholiker trifft, und bleibt seit dem Rückfall danach selbst zum Treffen da.",
     "{name} schließt als Hausmeister den Kellerraum auf, in dem sich die Gruppe der Anonymen Alkoholiker trifft, und fährt danach wieder zum Betriebshof.",
     None, "occupational_dual_role"),
    (3, "easy", "health:cancer", ["Onkologen", "Chemotherapie"],
     "{name} hat übermorgen einen Termin beim Onkologen, um die nächste Chemotherapie zu besprechen.",
     "{name} hat für die Redaktion einen Termin beim Onkologen, um im Hintergrundgespräch die Kosten einer Chemotherapie zu besprechen.",
     None, "occupational"),
    (4, "easy", "health:hiv", ["antiretrovirale"],
     "Seit dem Befund nimmt {name} täglich {seine_ihre} antiretrovirale Medikation.",
     "{name} hat in der Klinikapotheke eine Schulung zur richtigen Lagerung antiretroviraler Präparate gehalten.",
     None, "occupational"),
    (5, "easy", "health:dialysis", ["Dialyse"],
     "{name} geht jeden Donnerstagnachmittag zur Dialyse.",
     "{name} fährt donnerstags die Nachbarin zur Dialyse und wartet währenddessen im Auto.",
     None, "third_party"),
    (6, "easy", "health:pregnancy", ["schwanger"],
     "{name} hat der Kita mitgeteilt, dass {er_sie} im sechsten Monat schwanger ist.",
     "{name} hat der Kita mitgeteilt, dass die Gruppenleiterin im sechsten Monat schwanger ist und ab Mai vertreten wird.",
     "female", "third_party"),
    (7, "easy", "health:epilepsy", ["Epilepsie-Notfallausweis"],
     "{name} trägt seit dem Frühjahr einen Epilepsie-Notfallausweis bei sich.",
     "{name} hat in der Erste-Hilfe-Schulung erklärt, wozu ein Epilepsie-Notfallausweis dient.",
     None, "pedagogical"),
    (8, "easy", "health:disability", ["Schwerbehindertenausweis", "Rollstuhl"],
     "{name} hat nach dem Unfall einen Schwerbehindertenausweis beantragt und nutzt seitdem einen Rollstuhl.",
     "{name} erklärt am Bürgerschalter täglich, wie man einen Schwerbehindertenausweis beantragt und wo es Zuschüsse für einen Rollstuhl gibt.",
     None, "occupational"),
    (9, "medium", "health:cancer", ["Bestrahlungstermine"],
     "{name} koordiniert in der Radiologie die Bestrahlungstermine und fällt nun selbst einige Wochen aus, weil die eigenen vor kurzem begonnen haben.",
     "{name} koordiniert in der Radiologie die Bestrahlungstermine und musste dem Team die neue Wochenplanung mitteilen.",
     None, "occupational_dual_role"),
    (10, "medium", "health:disability", ["Beinprothese", "Reha-Programm"],
     "{name} passt in der Werkstatt jede Beinprothese an und trägt seit dem eigenen Unfall selbst eine, weshalb {er_sie} am wöchentlichen Reha-Programm teilnimmt.",
     "{name} passt in der Werkstatt jede Beinprothese an und begleitet die Patienten anschließend im wöchentlichen Reha-Programm.",
     None, "occupational_dual_role"),
    (11, "medium", "health:diabetes", ["Blutzucker", "Insulin"],
     "{name} erklärt in der Apotheke mehrmals täglich, wie man den Blutzucker misst, und verlässt zwischendurch kurz den Verkaufsraum, um sich selbst Insulin zu spritzen.",
     "{name} erklärt in der Apotheke mehrmals täglich, wie man den Blutzucker misst und Insulin richtig lagert.",
     None, "occupational_dual_role"),
    (12, "medium", "health:pregnancy", ["Mutterschutz", "Geburtstermin"],
     "{name} berechnet in der Personalabteilung den Mutterschutz der Kolleginnen und hat dort inzwischen den eigenen Geburtstermin gemeldet.",
     "{name} berechnet in der Personalabteilung den Mutterschutz der Kolleginnen, sobald ein Geburtstermin gemeldet wird.",
     "female", "occupational_dual_role"),
    (13, "medium", "health:cancer", ["krankgeschrieben", "Chemo-Sitzung"],
     "{name} trägt in der Praxissoftware ein, wer krankgeschrieben ist, und steht seit der eigenen Chemo-Sitzung selbst auf dieser Liste.",
     "{name} trägt in der Praxissoftware ein, wer krankgeschrieben ist und wann die nächste Chemo-Sitzung angesetzt wird.",
     None, "occupational_dual_role"),
    (14, "medium", "health:addiction", ["Entwöhnungsklinik", "keinen Alkohol"],
     "{name} trinkt seit der Entwöhnungsklinik keinen Alkohol mehr und lehnt auf Feiern jedes angebotene Glas ab.",
     "{name} beliefert die Entwöhnungsklinik mit Getränken und lädt dort seit Jahren keinen Alkohol ab.",
     None, "occupational"),
    (15, "medium", "health:epilepsy", ["Anfall"],
     "{name} hat um einen Platz nah am Ausgang gebeten, falls während des Meetings ein Anfall auftritt.",
     "{name} hat im Notfallplan festgehalten, welche Reihe frei bleiben muss, falls bei einem Gast ein Anfall auftritt.",
     None, "occupational"),
    (16, "medium", "health:autoimmune", ["Autoimmunerkrankung"],
     "In der Personalakte von {name} ist vermerkt, dass wegen der Autoimmunerkrankung regelmäßig Medikamente eingenommen und direkte Sonne gemieden werden.",
     "In der Personalakte von {name} ist vermerkt, dass für die Masterarbeit über eine seltene Autoimmunerkrankung anonymisierte Akten ausgewertet werden dürfen.",
     None, "pedagogical"),
    (17, "hard", "health:eating_disorder", ["stark abgenommen", "kaum noch etwas"],
     "{name} hat in letzter Zeit stark abgenommen und isst in der Kantine kaum noch etwas, was den Kollegen auffällt.",
     "{name} berichtet, dass der alte Hund stark abgenommen hat und aus dem Napf kaum noch etwas frisst.",
     None, "animal"),
    (18, "hard", "health:diabetes", ["kleines rundes Pflaster"],
     "{name} trägt seit kurzem ein kleines rundes Pflaster am Oberarm, das {er_sie} beim Sport immer wieder verdeckt.",
     "{name} trägt nach der Reiseimpfung noch ein kleines rundes Pflaster am Oberarm, das morgen abgenommen werden kann.",
     None, "benign_cause"),
    (19, "hard", "health:epilepsy", ["nah an der Tür", "schwindlig"],
     "{name} bat darum, in Besprechungen nah an der Tür zu sitzen, falls {ihm_ihr} plötzlich schwindlig wird.",
     "{name} bat darum, in dem stickigen Raum nah an der Tür zu sitzen, weil vielen Teilnehmenden dort sonst schwindlig wird.",
     None, "benign_cause"),
    (20, "hard", "health:mental_health", ["Therapiebeginn"],
     "{name} plant in der Praxis den Therapiebeginn der neuen Patienten und hält sich montags den Nachmittag für die eigene Sitzung frei.",
     "{name} plant in der Praxis den Therapiebeginn der neuen Patienten und hält sich montags den Nachmittag für die Dokumentation frei.",
     None, "occupational_dual_role"),
    (21, "hard", "health:mental_health", ["geröteten Augen"],
     "Kollegen bemerkten, dass {name} nach der Mittagspause oft mit geröteten Augen zurückkommt und sehr in sich gekehrt wirkt.",
     "Kollegen bemerkten, dass {name} nach der Mittagspause im Garten oft mit geröteten Augen zurückkommt, weil der Pollenflug gerade sehr stark ist.",
     None, "benign_cause"),
    (22, "hard", "health:addiction", ["nur Wasser", "wechselt schnell das Thema"],
     "{name} trinkt auf Feiern grundsätzlich nur Wasser und wechselt schnell das Thema, wenn jemand nach dem Grund fragt.",
     "{name} trinkt auf Feiern nur Wasser, weil danach noch die Nachtschicht ansteht, und wechselt schnell das Thema, um nicht über Dienstpläne zu reden.",
     None, "benign_cause"),
    (23, "hard", "health:chronic_fatigue", ["Energie im Tagesverlauf stark"],
     "{name} hat gebeten, Termine möglichst am Vormittag zu legen, da die Energie im Tagesverlauf stark nachlässt.",
     "{name} hat in der Studie gezeigt, dass bei allen Teilnehmenden die Energie im Tagesverlauf stark nachlässt.",
     None, "pedagogical"),
    (24, "hard", "health:mental_health", ["zitterten", "Tablettenschachtel"],
     "Im Meeting fiel auf, dass die Hände von {name} leicht zitterten, während {er_sie} eine kleine Tablettenschachtel in der Jackentasche verstaute.",
     "Nach dem Umzug fiel auf, dass die Hände von {name} vor Anstrengung zitterten, während {er_sie} die letzte Tablettenschachtel in die Reiseapotheke einräumte.",
     None, "benign_cause"),
    (25, "hard", "health:mental_health", ["größere Menschenansammlungen", "schwerfallen"],
     "{name} hat die Teilnahme an der Weihnachtsfeier abgesagt, weil größere Menschenansammlungen {ihm_ihr} zunehmend schwerfallen.",
     "{name} erklärte im Seminar, warum größere Menschenansammlungen vielen Betroffenen so schwerfallen.",
     None, "pedagogical"),
]

ETHNICITY_PAIRS = [
    (1, "easy", "ethnicity:religious_practice_islam", ["fünfmal am Tag"],
     "{name} betet fünfmal am Tag und nimmt sich dafür während der Arbeitszeit eine kurze Pause.",
     "{name} hat im Ethikunterricht erklärt, warum gläubige Muslime fünfmal am Tag beten.",
     None, "pedagogical"),
    (2, "easy", "ethnicity:religious_practice_hindu", ["Diwali"],
     "{name} feiert jedes Jahr im Herbst das Lichterfest Diwali mit der ganzen Familie.",
     "{name} begleitet die Nachbarsfamilie im Herbst zum Lichterfest Diwali, weil sie eine Fahrgelegenheit gesucht hat.",
     None, "third_party"),
    (3, "easy", "ethnicity:religious_practice_islam", ["kein Schweinefleisch", "halal"],
     "{name} isst aus religiösen Gründen kein Schweinefleisch und bevorzugt halal-zertifizierte Gerichte.",
     "{name} hat für die Kantine festgelegt, dass montags kein Schweinefleisch angeboten wird und ein halal-zertifizierter Lieferant gesucht wird.",
     None, "occupational"),
    (4, "easy", "ethnicity:language_turkish", ["Türkisch"],
     "{name} spricht zu Hause hauptsächlich Türkisch mit den Eltern.",
     "{name} spricht im Abendkurs ausschließlich Türkisch, um sich auf die Prüfung vorzubereiten.",
     None, "benign_cause"),
    (5, "easy", "ethnicity:religious_practice_islam", ["Freitagsgebet", "Moschee", "Kopftuch"],
     "{name} fotografiert für die Lokalzeitung das Freitagsgebet in der Moschee und trägt dabei wie sonst auch das eigene Kopftuch.",
     "{name} fotografiert für die Lokalzeitung das Freitagsgebet in der Moschee und trägt dort aus Höflichkeit ein geliehenes Kopftuch.",
     "female", "occupational_dual_role"),
    (6, "easy", "ethnicity:cultural_practice_persian", ["Nouruz"],
     "{name} feiert im Frühling das Fest Nouruz mit traditionellem Gebäck.",
     "{name} hat im Kulturzentrum den Saal vermietet, in dem im Frühling Nouruz gefeiert wird.",
     None, "occupational"),
    (7, "easy", "ethnicity:religious_practice_orthodox", ["griechisch-orthodoxe Gemeinde"],
     "{name} besucht sonntags regelmäßig die griechisch-orthodoxe Gemeinde in der Stadt.",
     "{name} hat als Malermeister sonntags die Räume für die griechisch-orthodoxe Gemeinde gestrichen.",
     None, "occupational"),
    (8, "easy", "ethnicity:religious_practice_jewish", ["Hebräisch", "Schabbat"],
     "{name} lernt seit der Kindheit Hebräisch und liest daraus beim Schabbat vor.",
     "{name} unterrichtet an der Volkshochschule Hebräisch und erklärt dort auch, wie der Schabbat begangen wird.",
     None, "occupational"),
    (9, "medium", "ethnicity:cultural_practice_polish", ["Piroggen", "Polen"],
     "Beim Betriebsfest brachte {name} traditionelle Piroggen mit, die die Großmutter aus Polen {ihm_ihr} beigebracht hat.",
     "Beim Betriebsfest brachte {name} Piroggen mit, die es im Supermarkt gerade als Importware aus Polen im Angebot gab.",
     None, "benign_cause"),
    (10, "medium", "ethnicity:religious_practice_islam", ["freitagnachmittags", "Moschee"],
     "{name} nimmt sich freitagnachmittags mehr Zeit für die Mittagspause, um in die nahegelegene Moschee zu gehen.",
     "{name} liefert freitagnachmittags den Mittagstisch an die nahegelegene Moschee aus.",
     None, "occupational"),
    (11, "medium", "ethnicity:religious_practice_jewish", ["jüdischen Neujahrsfest", "Apfel und Honig"],
     "Zum jüdischen Neujahrsfest lud {name} die Kollegen zu einem Essen mit Apfel und Honig ein.",
     "Für die Ausstellung zum jüdischen Neujahrsfest stellte {name} eine Vitrine mit Apfel und Honig zusammen.",
     None, "occupational"),
    (12, "medium", "ethnicity:religious_practice_islam", ["Ramadan"],
     "{name} hat während des Ramadan gebeten, Teammeetings vor Sonnenuntergang zu legen.",
     "{name} hat den Projektplan angepasst, weil mehrere Abgabetermine in den Ramadan fallen und Teile des Teams dann früher Feierabend machen.",
     None, "logistical"),
    (13, "medium", "ethnicity:cultural_practice_indian", ["Sari", "Mumbai"],
     "{name} trägt bei Familienfeiern einen Sari, den {ihm_ihr} die Tante aus Mumbai geschickt hat.",
     "{name} hat für die Theaterproduktion einen Sari besorgt, der eigens aus Mumbai geliefert wurde.",
     None, "occupational"),
    (14, "medium", "ethnicity:cultural_practice_chinese", ["chinesische Neujahrsfest", "rote Umschläge"],
     "{name} feiert Ende Januar mit der Familie das chinesische Neujahrsfest und bringt danach oft rote Umschläge mit.",
     "Die Nachbarsfamilie feiert das chinesische Neujahrsfest und hat {name} dabei rote Umschläge für die eigenen Kinder überreicht.",
     None, "third_party"),
    (15, "medium", "ethnicity:religious_practice_jewish", ["samstags kein Licht"],
     "Den Kollegen wurde erklärt, dass in der Familie von {name} samstags kein Licht angeschaltet und nicht gearbeitet wird.",
     "Den Kollegen wurde erklärt, dass im Bürogebäude, das {name} verwaltet, samstags kein Licht brennt, um Strom zu sparen.",
     None, "logistical"),
    (16, "hard", "ethnicity:religious_practice_islam", ["Schinkenhäppchen", "Wein"],
     "{name} ließ auf der Feier sowohl die Schinkenhäppchen als auch den Wein aus und griff nur beim Gebäck zu.",
     "{name} ließ auf der Feier die Schinkenhäppchen und den Wein aus, weil {er_sie} vegetarisch isst und anschließend noch fahren musste.",
     None, "benign_cause"),
    (17, "hard", "ethnicity:cultural_practice_orthodox|ethnicity:cultural_practice_chinese",
     ["im Januar ein anderes großes Fest"],
     "In der Weihnachtszeit erklärte {name} den Kollegen, dass in der Familie stattdessen im Januar ein anderes großes Fest gefeiert wird.",
     "In der Weihnachtszeit kündigte {name} an, dass die Firma im Januar ein anderes großes Fest für alle Standorte ausrichtet.",
     None, "logistical"),
    (18, "hard", "ethnicity:religious_practice_jewish", ["Samstag für die Familie reserviert"],
     "{name} bat darum, den Termin zu verschieben, da der Samstag für die Familie reserviert sei.",
     "{name} teilte mit, dass im Ferienhaus der Samstag für die Familie reserviert sei und Gäste erst ab Sonntag kommen können.",
     None, "benign_cause"),
    (19, "hard", "ethnicity:cultural_practice_east_asian|ethnicity:cultural_practice_middle_eastern",
     ["vor der Wohnungstür"],
     "{name} legt die Schuhe automatisch vor der Wohnungstür ab, bevor Besuch hereingebeten wird.",
     "{name} stellt die schmutzigen Wanderstiefel vor der Wohnungstür ab, weil der Teppich frisch gereinigt wurde.",
     None, "benign_cause"),
    (20, "hard", "ethnicity:cultural_practice_middle_eastern", ["Kreuzkümmel und Sumach"],
     "{name} bringt zur Mittagspause oft ein Gericht mit, das nach Kreuzkümmel und Sumach duftet.",
     "{name} hat für den Kochkurs Kreuzkümmel und Sumach eingekauft, weil die Teilnehmenden ein neues Rezept ausprobieren.",
     None, "occupational"),
]

# Mixed pairs carry a direct identifier in BOTH members, so the identifier
# cannot predict the sensitive-attribute label either.
MIXED_PAIRS = [
    (1, "easy", "health:diabetes", ["spritzt", "Insulin"],
     "{name} (Kundennummer {kdnr}) spritzt in der Tierarztpraxis den Katzen das Insulin und seit der eigenen Diagnose auch sich selbst.",
     {"name": "PERSON", "kdnr": "ID_KUNDENNUMMER"},
     "{name} (Kundennummer {kdnr}) spritzt in der Tierarztpraxis den Katzen das Insulin, wenn die Besitzer es nicht schaffen.",
     {"name": "PERSON", "kdnr": "ID_KUNDENNUMMER"}, None, "occupational_dual_role"),
    (2, "easy", "health:dialysis", ["Dialyse"],
     "Die Mitarbeiter-ID von {name} lautet {maid}; {er_sie} geht donnerstags zur Dialyse.",
     {"name": "PERSON", "maid": "ID_MITARBEITER"},
     "Die Mitarbeiter-ID von {name} lautet {maid}; {er_sie} fährt donnerstags den Fahrdienst zur Dialyse.",
     {"name": "PERSON", "maid": "ID_MITARBEITER"}, None, "occupational"),
    (3, "easy", "ethnicity:religious_practice_islam", ["fünfmal am Tag"],
     "{name} wohnt in der {address} und betet fünfmal am Tag.",
     {"name": "PERSON", "address": "ADDRESS"},
     "{name} wohnt in der {address} und kontrolliert dort fünfmal am Tag den Wasserstand der Baustelle.",
     {"name": "PERSON", "address": "ADDRESS"}, None, "occupational"),
    (4, "easy", "health:cancer", ["Chemotherapie"],
     "Kontaktieren Sie {name} unter {email} bezüglich der bevorstehenden Chemotherapie.",
     {"name": "PERSON", "email": "EMAIL"},
     "Kontaktieren Sie {name} unter {email}; {er_sie} begleitet den Vater zur Chemotherapie und ist tagsüber schlecht erreichbar.",
     {"name": "PERSON", "email": "EMAIL"}, None, "third_party"),
    (5, "easy", "ethnicity:religious_practice_islam", ["Zuckerfest"],
     "{name}, erreichbar unter {phone}, feiert jedes Jahr das Zuckerfest mit der Familie.",
     {"name": "PERSON", "phone": "PHONE"},
     "{name}, erreichbar unter {phone}, koordiniert das Straßenfest, das dieses Jahr auf das Zuckerfest fällt.",
     {"name": "PERSON", "phone": "PHONE"}, None, "occupational"),
    (6, "medium", "ethnicity:religious_practice_islam", ["Ramadan"],
     "Die Sozialversicherungsnummer {svnr} gehört zu {name}, {der_die} die Sonderschichten während des Ramadan eingeplant hat und tagsüber selbst fastet.",
     {"svnr": "ID_SVNR", "name": "PERSON"},
     "Die Sozialversicherungsnummer {svnr} gehört zu {name}, {der_die} die Sonderschichten während des Ramadan eingeplant hat.",
     {"svnr": "ID_SVNR", "name": "PERSON"}, None, "occupational_dual_role"),
    (7, "medium", "health:disability", ["Reha-Programm"],
     "Die IBAN {iban} gehört zu {name}, {der_die} seit Januar an einem Reha-Programm teilnimmt.",
     {"iban": "ID_IBAN", "name": "PERSON"},
     "Die IBAN {iban} gehört zu {name}, {der_die} die Rechnungen für das Reha-Programm der Klinik begleicht.",
     {"iban": "ID_IBAN", "name": "PERSON"}, None, "occupational"),
    (8, "hard", "health:eating_disorder", ["stark abgenommen", "kaum noch etwas"],
     "{der_die_kunde} – man schreibt {seinen_ihren} Namen {name_spaced} – hat stark abgenommen und isst kaum noch etwas.",
     {"name_spaced": "PERSON"},
     "{der_die_kunde} – man schreibt {seinen_ihren} Namen {name_spaced} – meldet, dass der Hund stark abgenommen hat und kaum noch etwas frisst.",
     {"name_spaced": "PERSON"}, None, "animal"),
    (9, "hard", "health:mental_health", ["zitterten"],
     "Wer unter der Nummer {phone_spelled} angerufen hat, war {name}, {dessen_deren} Hände im Gespräch leicht zitterten.",
     {"phone_spelled": "PHONE", "name": "PERSON"},
     "Wer unter der Nummer {phone_spelled} angerufen hat, war {name}; im Hintergrund zitterten die Scheiben vom Presslufthammer.",
     {"phone_spelled": "PHONE", "name": "PERSON"}, None, "benign_cause"),
    (10, "hard", "health:mental_health", ["geröteten Augen"],
     "KFZ: {kfz_spelled} — das Fahrzeug der Person, die nach der Mittagspause oft mit geröteten Augen zurückkommt.",
     {"kfz_spelled": "ID_KFZ"},
     "KFZ: {kfz_spelled} — das Fahrzeug des Lackierbetriebs, dessen Mitarbeitende bei starker Sonne oft mit geröteten Augen zurückkommen.",
     {"kfz_spelled": "ID_KFZ"}, None, "occupational"),
]

# ============================================================
# NON-PAIRED ROWS (no cue vocabulary at all)
# ============================================================

EXPLICIT_TEMPLATES = [
    (1, "easy", "{name} hat die Kundennummer {kdnr}.",
     {"name": "PERSON", "kdnr": "ID_KUNDENNUMMER"}),
    (2, "easy", "Die Sozialversicherungsnummer von {name} lautet {svnr}.",
     {"name": "PERSON", "svnr": "ID_SVNR"}),
    (3, "easy", "Bitte senden Sie die Unterlagen an {email}.",
     {"email": "EMAIL"}),
    (4, "easy", "{name}{gen_suffix} Wohnsitz befindet sich seit dem Umzug in der {address}.",
     {"name": "PERSON", "address": "ADDRESS"}),
    (5, "easy", "Bitte überweisen Sie den Betrag auf das Konto mit der IBAN {iban}.",
     {"iban": "ID_IBAN"}),
    (6, "medium", "Nachdem die Rechnung storniert wurde, bat {name} darum, künftig alle Schreiben an die {address} zu schicken, da unter der Kundennummer {kdnr} ein Fehler aufgetreten war.",
     {"name": "PERSON", "address": "ADDRESS", "kdnr": "ID_KUNDENNUMMER"}),
    (7, "medium", "{anrede} {name}, geboren in {city}, kann unter {email} oder telefonisch unter {phone} erreicht werden.",
     {"name": "PERSON", "email": "EMAIL", "phone": "PHONE"}),
    (8, "medium", "Die Personalabteilung bestätigte, dass {mitarbeiter_in} {name} mit der Mitarbeiter-ID {maid} am Montag {seinen_ihren} neuen Vertrag unterschreiben wird.",
     {"name": "PERSON", "maid": "ID_MITARBEITER"}),
    (9, "medium", "Die Steueridentifikationsnummer von {name} lautet {steuer_id}, die Krankenversichertennummer {kvnr}.",
     {"name": "PERSON", "steuer_id": "ID_STEUER", "kvnr": "ID_KVNR"}),
    (10, "hard", "{name_initials} bestätigte per E-Mail (Adresse endet auf {email_domain}) die neue Kundennummer; die Ziffern wurden einzeln durchgegeben: {kdnr_spelled}.",
     {"name_initials": "PERSON", "email_domain": "EMAIL", "kdnr_spelled": "ID_KUNDENNUMMER"}),
]

CONTROL_TEMPLATES = [
    (1, "{name} hat das Meeting heute um eine halbe Stunde verschoben."),
    (2, "Der Quartalsbericht muss bis Freitag fertiggestellt werden."),
    (3, "In {city} soll im nächsten Jahr eine neue Fahrradbrücke gebaut werden."),
    (4, "{name} hat sich für ein neues Notebook mit mehr Arbeitsspeicher entschieden."),
    (5, "Am Wochenende regnet es voraussichtlich in weiten Teilen des Landes."),
    (6, "{name} bringt am Montag den Projektentwurf zur Abstimmung mit."),
    (7, "Die Kantine bietet ab nächster Woche ein weiteres warmes Gericht an."),
    (8, "{name} hat den Drucker im zweiten Stock repariert."),
    (9, "Der Zug nach {city} hat heute zehn Minuten Verspätung."),
    (10, "{name} organisiert das Sommerfest der Abteilung in diesem Jahr."),
    (11, "{name} ist gestern einen Marathon in {city} gelaufen."),
    (12, "Das Team plant, die Präsentation am Mittwoch zu üben."),
    (13, "{name} sammelt seit Jahren alte Schallplatten."),
    (14, "Der Konferenzraum im dritten Stock wird diese Woche renoviert."),
    (15, "{name} hat den Vortrag über Datenschutz sehr interessant gefunden."),
    (16, "Die neue Kaffeemaschine steht ab Montag in der Teeküche."),
    (17, "{name} hat die Ablage im Archiv neu sortiert."),
    (18, "Der Aufzug im Nebengebäude wird am Dienstag gewartet."),
    (19, "{name} nimmt im Herbst an einer Fortbildung zu Projektmanagement teil."),
    (20, "Die Fahrradständer vor dem Eingang werden im Frühjahr erneuert."),
]

# ============================================================
# CUE CONFIDENCE  (positives only)
# ============================================================

CUE_CONFIDENCE = {
    ("implicit_health", 18): 0.7, ("implicit_health", 19): 0.5,
    ("implicit_health", 20): 0.7, ("implicit_health", 21): 0.6,
    ("implicit_health", 22): 0.7, ("implicit_health", 23): 0.7,
    ("implicit_health", 24): 0.6, ("implicit_health", 25): 0.7,
    ("implicit_health", 17): 0.7,
    ("implicit_ethnicity", 16): 0.7, ("implicit_ethnicity", 17): 0.5,
    ("implicit_ethnicity", 18): 0.7, ("implicit_ethnicity", 19): 0.5,
    ("implicit_ethnicity", 20): 0.6,
    ("mixed", 8): 0.7, ("mixed", 9): 0.6, ("mixed", 10): 0.6,
}
DEFAULT_CUE_CONFIDENCE = 0.9

TEMPLATE_NOTES = {
    ("implicit_health", 18): "ambiguous cue: a small round upper-arm patch may also indicate a nicotine patch or hormonal contraception, not only a CGM sensor",
    ("implicit_health", 19): "low-confidence cue: 'schwindlig' is weaker than 'Anfall'",
    ("implicit_health", 24): "dual reading: mental_health (tremor + concealment), not Parkinson's",
    ("implicit_ethnicity", 17): "dual reading: January festival — Orthodox Christmas or Lunar New Year",
    ("implicit_ethnicity", 19): "dual reading: removing shoes — East Asian or Middle Eastern practice",
    ("mixed", 10): "no PERSON span; identifier is the vehicle, cue is about an unnamed person",
}


def next_row_seed(base_offset, idx):
    return MASTER_SEED * 100000 + base_offset * 1000 + idx


# ============================================================
# POOL ASSIGNMENT
# ============================================================

def assign_group_splits(all_rows):
    """Dataset A is a single sealed evaluation pool.

    There is no train split because nothing is fitted on these sentences: the
    LoRA run uses Dataset B, whose cue vocabulary is disjoint by construction
    (see reserved_cues.txt). There is no dev split because prompt development
    and debugging happen on Dataset B's dev slice. Holding sentences back here
    would shrink the reported n without protecting anything.

    group_key is retained: it still records pair membership, which pair-level
    scoring needs downstream.
    """
    for row in all_rows:
        row["split"] = "test"


# ============================================================
# GENERATION
# ============================================================

sentences_rows, spans_rows, problems = [], [], []

DIRECT_ID_KINDS = {"PERSON", "ID_KUNDENNUMMER", "ID_SVNR", "ID_MITARBEITER", "EMAIL",
                   "PHONE", "ADDRESS", "ID_STEUER", "ID_IBAN", "ID_KFZ", "ID_KVNR"}


def add_row(sid, category, difficulty, group_key, pair_id, polarity, text, id_spans,
            implicit_attr, seed, cue_spans=None, decoy_spans=None, negative_type="",
            notes="", name_origin=""):
    cue_spans = list(cue_spans or [])
    decoy_spans = list(decoy_spans or [])
    redaction_spans = list(id_spans) + cue_spans
    all_spans = redaction_spans + decoy_spans
    assert_disjoint(sid, all_spans)

    if not any(sp[0] == "PERSON" for sp in all_spans):
        name_origin = ""

    redacted = generate_redacted_text(text, redaction_spans)
    conf = CUE_CONFIDENCE.get((category, pair_id), DEFAULT_CUE_CONFIDENCE) if pair_id else ""

    sentences_rows.append({
        "id": sid, "category": category, "difficulty": difficulty,
        "group_key": group_key,
        "pair_id": f"{category}_{pair_id:02d}" if pair_id else "",
        "polarity": polarity,
        "negative_type": negative_type,
        "text": text,
        "name_origin": name_origin,
        "name_origin_marked": bool(name_origin) and name_origin != "german",
        "implicit_attribute": (implicit_attr or "").split("|")[0],
        "implicit_attribute_secondary": "|".join((implicit_attr or "").split("|")[1:]),
        "n_spans": len(all_spans),
        "n_direct_id_spans": sum(1 for s in all_spans if s[0] in DIRECT_ID_KINDS),
        "n_implicit_cue_spans": len(cue_spans),
        "n_decoy_cue_spans": len(decoy_spans),
        "has_direct_identifier": any(s[0] in DIRECT_ID_KINDS for s in all_spans),
        "has_sensitive_attribute": bool(implicit_attr),
        "cue_confidence": conf if implicit_attr else "",
        "redacted_text": redacted,
        "source": "synthetic_minimal_pairs_de_v3_1",
        "seed": seed, "split": "", "notes": notes,
    })

    for i, (kind, sp_text, start, end) in enumerate(sorted(all_spans, key=lambda s: s[2]), 1):
        spans_rows.append({
            "sentence_id": sid, "span_id": f"{sid}_s{i:02d}",
            "text": sp_text, "kind": kind, "start": start, "end": end,
            "confidence": (conf if kind == "IMPLICIT_CUE" else
                           (0.0 if kind == "DECOY_CUE" else 1.0)),
            "source": "ground_truth",
        })


def build_cues(sid, category, pair_id, text, placed_spans, cue_keywords, kind):
    protected = [(s[2], s[3]) for s in placed_spans]
    spans, probs = find_keyword_spans(text, cue_keywords, protected, kind)
    for p in probs:
        problems.append(f"{sid} ({category}/{pair_id}): {p} | {text}")
    if len(spans) != len(cue_keywords):
        problems.append(f"{sid} ({category}/{pair_id}): "
                        f"{len(spans)}/{len(cue_keywords)} keyword spans | {text}")
    return spans


origin_counter = {}


def marked_flag(category):
    n = origin_counter.get(category, 0)
    origin_counter[category] = n + 1
    return n % 2 == 0


def emit_pair(category, base_offset, pair_id, difficulty, attr, kws,
              pos_tpl, pos_map, neg_tpl, neg_map, req_gender, neg_type, use_corpus):
    """Both members share one name and one gender."""
    seed = next_row_seed(base_offset, pair_id)
    name_str, gender, origin = pick_name(seed, marked_flag(category), gender=req_gender)
    note = TEMPLATE_NOTES.get((category, pair_id), "")

    for polarity, tpl, span_map in (("positive", pos_tpl, pos_map),
                                    ("negative", neg_tpl, neg_map)):
        row_seed = seed * 10 + (1 if polarity == "positive" else 2)
        sid = f"de_{category}_{pair_id:02d}_{'pos' if polarity == 'positive' else 'neg'}"
        resolved = resolve_german_pronouns(tpl, gender)
        if use_corpus:
            values = build_derived_values(Corpus(row_seed), name_str)
        else:
            values = {"name": name_str, "gen_suffix": genitive_suffix(name_str)}
        text, id_spans = render(resolved, values, span_map)

        if polarity == "positive":
            cues = build_cues(sid, category, pair_id, text, id_spans, kws, "IMPLICIT_CUE")
            n = note
            if neg_type == "occupational_dual_role":
                n = ((note + "; ") if note else "") + (
                    "dual-role positive: the subject is also the professional, so "
                    "occupational framing does not predict the label")
            add_row(sid, category, difficulty, f"pair_{pair_id:02d}", pair_id, polarity,
                    text, id_spans, attr, row_seed, cue_spans=cues,
                    name_origin=origin, notes=n)
        else:
            decoys = build_cues(sid, category, pair_id, text, id_spans, kws, "DECOY_CUE")
            n = (f"minimal-pair negative of {category}_{pair_id:02d}; "
                 f"same cue vocabulary, non-sensitive cause ({neg_type})")
            add_row(sid, category, difficulty, f"pair_{pair_id:02d}", pair_id, polarity,
                    text, id_spans, None, row_seed, decoy_spans=decoys,
                    negative_type=neg_type, name_origin=origin, notes=n)


# ---- Implicit health: 25 pairs = 50 rows ----
for pid, diff, attr, kws, pos_t, neg_t, g, ntype in HEALTH_PAIRS:
    emit_pair("implicit_health", 2, pid, diff, attr, kws,
              pos_t, {"name": "PERSON"}, neg_t, {"name": "PERSON"},
              g, ntype, use_corpus=False)

# ---- Implicit ethnicity: 20 pairs = 40 rows ----
for pid, diff, attr, kws, pos_t, neg_t, g, ntype in ETHNICITY_PAIRS:
    emit_pair("implicit_ethnicity", 3, pid, diff, attr, kws,
              pos_t, {"name": "PERSON"}, neg_t, {"name": "PERSON"},
              g, ntype, use_corpus=False)

# ---- Mixed: 10 pairs = 20 rows ----
for pid, diff, attr, kws, pos_t, pos_m, neg_t, neg_m, g, ntype in MIXED_PAIRS:
    emit_pair("mixed", 5, pid, diff, attr, kws,
              pos_t, pos_m, neg_t, neg_m, g, ntype, use_corpus=True)

# ---- Explicit: 10 templates x 2 = 20 rows ----
idx = 0
for template_id, difficulty, template, span_map in EXPLICIT_TEMPLATES:
    for _ in range(2):
        idx += 1
        seed = next_row_seed(1, idx)
        c = Corpus(seed)
        name_str, gender, origin = pick_name(seed, marked_flag("explicit"))
        resolved = resolve_german_pronouns(template, gender)
        values = build_derived_values(c, name_str)
        text, spans = render(resolved, values, span_map)
        add_row(f"de_explicit_{idx:03d}", "explicit", difficulty,
                f"tmpl_{template_id:02d}", None, "negative", text, spans, None, seed,
                negative_type="no_cue_vocabulary", name_origin=origin,
                notes="direct identifiers only; no sensitive attribute")

# ---- Control: 20 rows ----
idx = 0
for template_id, template in CONTROL_TEMPLATES:
    idx += 1
    seed = next_row_seed(4, idx)
    c = Corpus(seed)
    name_str, gender, origin = pick_name(seed, marked_flag("control"))
    resolved = resolve_german_pronouns(template, gender)
    values = {"name": name_str, "city": c.city()}
    text, spans = render(resolved, values, {"name": "PERSON"})
    add_row(f"de_control_{idx:03d}", "control", "none",
            f"tmpl_{template_id:02d}", None, "negative", text, spans, None, seed,
            negative_type="no_cue_vocabulary", name_origin=origin,
            notes="benign; no cue vocabulary, direct identifiers still labelled")

assign_group_splits(sentences_rows)

# ============================================================
# SELF-VALIDATION
# ============================================================

text_of = {r["id"]: r["text"] for r in sentences_rows}
row_of = {r["id"]: r for r in sentences_rows}
errors = []

for s in spans_rows:
    t = text_of[s["sentence_id"]]
    if t[s["start"]:s["end"]] != s["text"]:
        errors.append(f"offset mismatch: {s['span_id']}")

by_sent = defaultdict(list)
for s in spans_rows:
    by_sent[s["sentence_id"]].append(s)
for sid, sp in by_sent.items():
    sp = sorted(sp, key=lambda x: x["start"])
    for a, b in zip(sp, sp[1:]):
        if b["start"] < a["end"]:
            errors.append(f"overlap: {sid} {a['kind']}/{b['kind']}")

for r in sentences_rows:
    for s in by_sent[r["id"]]:
        if s["kind"] == "PERSON" and " " in s["text"] and s["text"] in r["redacted_text"]:
            errors.append(f"redaction leak: {r['id']}")

# retained as a guard: fires only if a split is ever reintroduced incorrectly
grp_splits = defaultdict(set)
for r in sentences_rows:
    grp_splits[(r["category"], r["group_key"])].add(r["split"])
for k, v in grp_splits.items():
    if len(v) > 1:
        errors.append(f"group leakage: {k} -> {sorted(v)}")

annotated = {s["text"] for s in spans_rows if s["kind"] == "PERSON" and " " in s["text"]}
for r in sentences_rows:
    labelled = {s["text"] for s in by_sent[r["id"]] if s["kind"] == "PERSON"}
    for n in [n for n in annotated if n in r["text"]]:
        if n not in labelled:
            errors.append(f"unlabelled name: {r['id']} -> {n}")

# --- pair integrity ---
pairs = defaultdict(list)
for r in sentences_rows:
    if r["pair_id"]:
        pairs[r["pair_id"]].append(r)
for pid, rows in pairs.items():
    if len(rows) != 2:
        errors.append(f"pair {pid} has {len(rows)} members")
        continue
    pol = sorted(r["polarity"] for r in rows)
    if pol != ["negative", "positive"]:
        errors.append(f"pair {pid} polarity {pol}")
    if len({r["name_origin"] for r in rows}) != 1:
        errors.append(f"pair {pid} name origin differs across members")
    for r in rows:
        if r["polarity"] == "negative":
            if r["has_sensitive_attribute"] or r["n_implicit_cue_spans"]:
                errors.append(f"pair {pid} negative carries a cue annotation")
            if not r["n_decoy_cue_spans"]:
                errors.append(f"pair {pid} negative has no decoy span")
        else:
            if not r["n_implicit_cue_spans"]:
                errors.append(f"pair {pid} positive has no cue span")

# --- CORE INVARIANT: every cue string occurs in both classes ---
LEXICON = sorted({kw for _, _, _, kws, *_ in HEALTH_PAIRS for kw in kws} |
                 {kw for _, _, _, kws, *_ in ETHNICITY_PAIRS for kw in kws} |
                 {kw for _, _, _, kws, *_ in MIXED_PAIRS for kw in kws})

kw_balance = {}
for kw in LEXICON:
    npos = sum(1 for r in sentences_rows
               if r["has_sensitive_attribute"] and boundary_find(r["text"], kw) != -1)
    nneg = sum(1 for r in sentences_rows
               if not r["has_sensitive_attribute"] and boundary_find(r["text"], kw) != -1)
    kw_balance[kw] = (npos, nneg)
    if npos == 0 or nneg == 0:
        errors.append(f"INVARIANT VIOLATED: {kw!r} occurs {npos} pos / {nneg} neg")

for p in problems:
    print("PROBLEM:", p, file=sys.stderr)
for e in errors:
    print("ERROR:", e, file=sys.stderr)

# ============================================================
# OUTPUT
# ============================================================

SENT_FIELDS = ["id", "category", "difficulty", "group_key", "pair_id", "polarity",
               "negative_type", "text", "name_origin", "name_origin_marked",
               "implicit_attribute", "implicit_attribute_secondary",
               "n_spans", "n_direct_id_spans", "n_implicit_cue_spans", "n_decoy_cue_spans",
               "has_direct_identifier", "has_sensitive_attribute", "cue_confidence",
               "redacted_text", "source", "seed", "split", "notes"]
SPAN_FIELDS = ["sentence_id", "span_id", "text", "kind", "start", "end", "confidence", "source"]

with open("sentences.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=SENT_FIELDS)
    w.writeheader()
    w.writerows(sentences_rows)

with open("spans.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=SPAN_FIELDS)
    w.writeheader()
    w.writerows(spans_rows)

# ------------------------------------------------------------------
# RESERVED VOCABULARY — the contract between Dataset A and Dataset B
# ------------------------------------------------------------------
# Every cue string below occurs in this evaluation set. Dataset B's generator
# MUST import reserved_cues.txt and assert that none of them appears in any
# training or dev sentence. If a cue appears in both, the fine-tuned model has
# seen the test vocabulary while the prompted baselines have not, and the
# fine-tuning result measures memorisation rather than generalisation.
#
# RESERVED_ATTRIBUTES never appear in B at all, so evaluation can be reported
# separately for attributes seen during fine-tuning and attributes never seen.
# That contrast is the direct test of whether the model learned contextual
# reasoning or merely the training cues.

RESERVED_ATTRIBUTES = [
    "health:chronic_fatigue",
    "health:autoimmune",
    "health:epilepsy",
    "ethnicity:cultural_practice_persian",
    "ethnicity:cultural_practice_indian",
    "ethnicity:religious_practice_orthodox",
]

with open("reserved_cues.txt", "w", encoding="utf-8") as f:
    f.write("# Reserved cue strings from Dataset A. Must NOT appear in Dataset B.\n")
    f.write("# Match with boundary_find(), not `in`: 'Wein' is a substring of\n")
    f.write("# 'Schweinefleisch' and a naive check would pass a real collision.\n")
    for kw in LEXICON:
        f.write(kw + "\n")

with open("reserved_attributes.txt", "w", encoding="utf-8") as f:
    f.write("# Attributes held out of Dataset B entirely (unseen-attribute eval).\n")
    for a in RESERVED_ATTRIBUTES:
        f.write(a + "\n")

known_attrs = {r["implicit_attribute"] for r in sentences_rows if r["implicit_attribute"]}
unknown = [a for a in RESERVED_ATTRIBUTES if a not in known_attrs]
assert not unknown, f"reserved attribute not present in dataset A: {unknown}"

# backwards compatibility for anything already reading the old filename
with open("cue_lexicon.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(LEXICON) + "\n")

# ============================================================
# BASELINES
# ============================================================


def keyword_baseline():
    """Strongest possible lexicon system: it is *given* the complete gold cue
    list, including every hard cue. Sentence level: flag a row if any cue string
    occurs. Span level: predict every occurrence as a cue span."""
    tp = fp = fn = tn = 0
    span_tp = span_fp = 0
    gold_cue_total = sum(1 for s in spans_rows if s["kind"] == "IMPLICIT_CUE")

    for r in sentences_rows:
        raw = [(boundary_find(r["text"], kw), kw) for kw in LEXICON]
        raw = sorted((s, s + len(kw), kw) for s, kw in raw if s != -1)
        hits, last_end = [], -1
        for s, e, kw in raw:
            if s >= last_end:
                hits.append((s, e, kw))
                last_end = e
        pred = bool(hits)
        gold = bool(r["has_sensitive_attribute"])
        if pred and gold:
            tp += 1
        elif pred and not gold:
            fp += 1
        elif not pred and gold:
            fn += 1
        else:
            tn += 1

        gold_spans = [(s["start"], s["end"]) for s in by_sent[r["id"]]
                      if s["kind"] == "IMPLICIT_CUE"]
        for start, end, _kw in hits:
            if any(start < ge and gs < end for gs, ge in gold_spans):
                span_tp += 1
            else:
                span_fp += 1

    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / len(sentences_rows)
    span_prec = span_tp / (span_tp + span_fp) if span_tp + span_fp else 0.0
    span_rec = span_tp / gold_cue_total if gold_cue_total else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, prec=prec, rec=rec, f1=f1, acc=acc,
                span_prec=span_prec, span_rec=span_rec)


def lexical_baseline():
    """Second shortcut probe: a bag-of-words classifier with pair-grouped CV.

    The keyword baseline tests one specific shortcut (gold cue lookup). This
    tests the general one: is there ANY lexical signal that separates the
    classes? Grouping by pair is essential -- without it a positive and its
    twin land in different folds and the score is inflated by near-duplicate
    leakage. A score near the majority class is the claim the minimal-pair
    design is meant to support.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GroupKFold, cross_val_score
        from sklearn.pipeline import make_pipeline
    except ImportError:
        return None

    X = [r["text"] for r in sentences_rows]
    y = [int(r["has_sensitive_attribute"]) for r in sentences_rows]
    groups = [f'{r["category"]}_{r["group_key"]}' for r in sentences_rows]
    pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=1),
                         LogisticRegression(max_iter=2000))
    acc = cross_val_score(pipe, X, y, groups=groups, cv=GroupKFold(n_splits=5),
                          scoring="accuracy")
    f1 = cross_val_score(pipe, X, y, groups=groups, cv=GroupKFold(n_splits=5),
                         scoring="f1")
    return dict(acc=acc.mean(), acc_sd=acc.std(), f1=f1.mean(), f1_sd=f1.std())


bl = keyword_baseline()
lex = lexical_baseline()
n_pos = sum(1 for r in sentences_rows if r["has_sensitive_attribute"])
n_rows = len(sentences_rows)
majority = max(n_pos, n_rows - n_pos) / n_rows

print(f"\nGenerated {n_rows} sentences, {len(spans_rows)} spans")
print(f"Problems: {len(problems)}   Validation errors: {len(errors)}")

print("\nEvaluation pool composition (single sealed pool, no split):")
print(f"  total n={n_rows}   pos={n_pos}   neg={n_rows - n_pos}")
for field in ("category", "difficulty"):
    print(f"  by {field}:")
    for k, c in sorted(Counter(r[field] for r in sentences_rows if r[field]).items()):
        print(f"    {k:24s} {c}")
print("  positives by difficulty:")
for k, c in sorted(Counter(r["difficulty"] for r in sentences_rows
                           if r["has_sensitive_attribute"]).items()):
    print(f"    {k:24s} {c}")

print("\nSpan kind distribution:")
for k, n in Counter(s["kind"] for s in spans_rows).most_common():
    print(f"  {k:24s} {n}")

print("\nNegative types (paired rows only):")
paired_neg = [r for r in sentences_rows if r["pair_id"] and r["polarity"] == "negative"]
for k, n in Counter(r["negative_type"] for r in paired_neg).most_common():
    print(f"  {k:24s} {n}   ({n / len(paired_neg):.0%})")

print("\n" + "=" * 62)
print("ORACLE KEYWORD BASELINE  (given the complete gold cue lexicon)")
print("=" * 62)
print(f"  sentence  precision = {bl['prec']:.3f}")
print(f"  sentence  recall    = {bl['rec']:.3f}")
print(f"  sentence  F1        = {bl['f1']:.3f}")
print(f"  sentence  accuracy  = {bl['acc']:.3f}   (majority class = {majority:.3f})")
print(f"  confusion tp={bl['tp']} fp={bl['fp']} fn={bl['fn']} tn={bl['tn']}")
print(f"  span      precision = {bl['span_prec']:.3f}")
print(f"  span      recall    = {bl['span_rec']:.3f}")
print(f"  -> accuracy over majority class: {bl['acc'] - majority:+.3f}")

print("\n" + "=" * 62)
print("LEXICAL BASELINE  (bag-of-words + LR, pair-grouped 5-fold CV)")
print("=" * 62)
if lex is None:
    print("  scikit-learn not installed; skipped")
else:
    print(f"  accuracy = {lex['acc']:.3f} (sd {lex['acc_sd']:.3f})   "
          f"majority class = {majority:.3f}")
    print(f"  F1       = {lex['f1']:.3f} (sd {lex['f1_sd']:.3f})")
    print(f"  -> accuracy over majority class: {lex['acc'] - majority:+.3f}")

print("\nCue lexicon balance (occurrences pos / neg) — worst 8:")
for kw, (p, n) in sorted(kw_balance.items(), key=lambda kv: -abs(kv[1][0] - kv[1][1]))[:8]:
    print(f"  {kw[:44]:44s} {p:2d} / {n:2d}")

print("\nAttribute coverage (R = reserved, held out of Dataset B):")
cov = Counter()
for r in sentences_rows:
    if r["implicit_attribute"]:
        cov[r["implicit_attribute"]] += 1
for a in sorted(cov):
    flag = "R" if a in RESERVED_ATTRIBUTES else " "
    print(f"  {flag} {a:52s} n={cov[a]:2d}")
n_reserved = sum(v for a, v in cov.items() if a in RESERVED_ATTRIBUTES)
print(f"\nReserved: {len(LEXICON)} cue strings, {len(RESERVED_ATTRIBUTES)} attributes "
      f"({n_reserved} positive sentences) held out of Dataset B.")

# --- confound diagnostic: can name origin alone predict the label? ---
def name_origin_check(rows, label):
    tab = Counter()
    for r in rows:
        if not r["name_origin"]:
            continue
        key = "marked" if r["name_origin_marked"] else "german"
        lab = "eth_cue" if r["implicit_attribute"].startswith("ethnicity") else "no_eth_cue"
        tab[(key, lab)] += 1
    a, b = tab[("marked", "eth_cue")], tab[("marked", "no_eth_cue")]
    c_, d = tab[("german", "eth_cue")], tab[("german", "no_eth_cue")]
    n_tot = a + b + c_ + d
    if n_tot == 0 or (a + b) == 0 or (c_ + d) == 0:
        print(f"  {label:22s} insufficient data")
        return
    maj = max(a + c_, b + d) / n_tot
    name_only = (max(a, b) + max(c_, d)) / n_tot
    print(f"  {label:22s} P(eth|marked)={a/(a+b):.3f}  P(eth|german)={c_/(c_+d):.3f}  "
          f"majority={maj:.3f}  name-only={name_only:.3f}  gain={name_only - maj:+.3f}")


print("\nName-origin confound check:")
name_origin_check(sentences_rows, "all")
for cat in sorted({r["category"] for r in sentences_rows}):
    name_origin_check([r for r in sentences_rows if r["category"] == cat], cat)

if errors:
    sys.exit(1)
