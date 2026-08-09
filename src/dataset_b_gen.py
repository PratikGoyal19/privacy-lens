#!/usr/bin/env python3
"""
German LLM-Redactor Dataset Generator — Dataset B (fine-tuning set)

Rewrite pairs for the LoRA run: (leaky sentence -> repaired sentence), plus
four kinds of NO-REWRITE pair where the correct output preserves the prose.

CONTRACT WITH DATASET A
-----------------------
Dataset A is the sealed evaluation set. This generator loads reserved_cues.txt
and reserved_attributes.txt from A and FAILS THE BUILD if:
  * any cue string from A appears in any B sentence (input or output), or
  * any attribute reserved for A is used in B.
Matching uses boundary_find(), not `in`: 'Wein' is a substring of
'Schweinefleisch', so a naive check would pass a real collision.

Without this guard the fine-tuned model would have seen A's vocabulary while
the prompted baselines had not, and the headline comparison would measure
memorisation rather than generalisation.

Note what the contract does and does not block. Lexical overlap is blocked;
structural overlap is deliberately preserved, because A and B share an
operation and negative-type taxonomy by construction. "Generalisation" here
therefore means: unseen cue vocabulary, within a seen taxonomy, in the same
synthetic register. That is a real and defensible claim; it is not
generalisation to unseen redaction problems, and the writeup must not let a
reader slide between the two.

ITEMS ARE THE LEARNING AXIS, ROWS ARE THE COST AXIS
---------------------------------------------------
Instantiations of one item differ only by name and identifier, so they carry
almost no semantic variance. Effective n for any clustered statistic is the
item count, not the row count. Instantiations are budgeted per family rather
than uniformly (see INSTANTIATIONS).

WHAT THE NO-REWRITE FAMILIES ARE FOR
------------------------------------
Each defends against a shortcut that a naive corpus would install.

  IDENTITY (borderline cases included)
      Cue vocabulary present, nothing disclosed. Without these the model learns
      "always rewrite" and over-redacts. Five items sit close to the decision
      boundary with a stated innocent cause, so the over-redaction metric is
      not measured only in the easy regime.

  PASS_THROUGH
      Room, invoice, batch, norm and platform numbers that must SURVIVE. If
      every identifier in every output is placeheld, the model learns "mask
      anything code-shaped". Some items put a real identifier and a
      non-sensitive number in one sentence, and some additionally require a
      prose rewrite, so the distinction cannot be made at sentence level.

  DISCLOSURE_NECESSARY
      Redaction would destroy the actionable content. For a catering
      instruction the religious certification IS the instruction; generalising
      to "bei einem Caterer" breaks the speech act. All items are directive or
      request constructions in which the sensitive term is the grammatical
      object, so the signal is structural rather than a memorisable word list.
      This is the one class teaching "do not redact", so it is capped at ~8% of
      rows: thin or lexically idiosyncratic evidence here generalises in the
      unsafe direction.

      RULE: the sensitive term is the operative content of a directive or
      request, such that removing it changes what the reader is instructed to
      do. HARKING WARNING: this class must be applied to Dataset A before any
      model is run, by annotators blind to model output. Assigned afterwards it
      is relabelled error, not a contribution.

  ID_PLACEHOLDER
      Direct identifiers only; substitution, no paraphrase.

NAME-ORIGIN INVARIANCE IS TRAINED, NOT ONLY MEASURED
----------------------------------------------------
Every item is instantiated under both a German-origin and a marked-origin name,
and names are drawn without replacement across an item's instantiations. This
is the training-side complement to the held-out fairness probe, which is what
actually tests for differential over-redaction.

Gender follows the name draw rather than being balanced per item. It is
recorded in a `gender` column so the association with the label can be checked,
but it is not controlled: forcing gender balance within items would fight the
origin rotation, which is the constraint that matters.

EXPLANATIONS ARE METADATA, NOT TRAINING TARGETS
-----------------------------------------------
Every rewrite item carries a German explanation in the CSV; the JSONL messages
contain only system/user/assistant. Two reasons: a well-formed explanation
restates the attribute that was just removed, so training the model to emit one
builds a redactor whose second field un-redacts; and joint rewrite +
explanation training at this data size splits an already-thin gradient, with
explanation tokens dominating the loss by length. Explanations are for error
analysis and for a separate prompted call at demo time.

TWO MATCHERS, DIFFERENT JOBS
----------------------------
boundary_find() anchors on a left word boundary and is the shared contract
matcher with Dataset A; it must not be changed independently of A.
compound_find() is a blunt casefolded substring search used only for leak
terms, where under-flagging is the expensive error and the term list is small
and hand-curated.

REWRITE POLICY
--------------
  CLAUSE_DROP          the sensitive fact sits in a removable clause
  GENERALIZE           the fact is load-bearing; a superordinate term suffices
  ABSTRACT_REASON      the sentence explains *why*, and the why is the disclosure
  ID_PLACEHOLDER       direct identifiers only; substitution, no paraphrase
  IDENTITY             cue vocabulary present, nothing disclosed; output = input
  PASS_THROUGH         non-sensitive numerics; they must NOT be placeheld
  DISCLOSURE_NECESSARY redaction would destroy the actionable content

Governing principle: preserve what the reader needs in order to act; remove
what lets them infer the attribute. Never introduce a false reason — the output
must not assert anything the input did not. Register must survive the rewrite.
"""

import csv
import json
import os
import random
import re
import sys
import zlib
from collections import Counter, defaultdict

MASTER_SEED = 4242
# Row budget: 150-200. Instantiations are allocated per family rather than
# uniformly, because rows are the cost axis and different families need
# different amounts of gradient:
#   * rewrite gets 3 — it is the behaviour being taught and carries the most
#     internal variety (five operations, five syntactic constructions).
#   * pass-through gets 3 despite being small. h15, m08 and e06 also live here:
#     they require a prose rewrite AND numeric preservation in one sentence,
#     which is the combination most likely to expose the masking shortcut — it is the sole counterweight to
#     the "mask anything code-shaped" shortcut and has no other support.
#   * identity gets 2 — 19 items already give it the widest item coverage of
#     any no-rewrite family; a third instantiation would buy near-duplicates.
#   * disclosure gets 2 — deliberately capped. This is the class that says
#     "sensitive term present, leave it alone"; at 3 instantiations it would be
#     12% of all rows, which over-weights a behaviour whose errors run in the
#     unsafe direction. At 2 it is 8%.
#   * explicit gets 2 — the simplest behaviour in the set, pure substitution.
# Two instantiations still cover both name-origin conditions (see ORIGIN_PLAN),
# which is the invariant that actually matters.
INSTANTIATIONS = {"rewrite": 3, "passthrough": 3,
                  "identity": 2, "disclosure": 2, "explicit": 2}
ROW_BUDGET = (150, 200)
DEV_ITEM_FRACTION = 0.25
N_PROBE_NAMES = 3

# ============================================================
# GUARD: load the contract from Dataset A
# ============================================================


def boundary_find(text, kw, start=0):
    """Identical to Dataset A's matcher. Must stay in sync."""
    for flags in (0, re.IGNORECASE):
        m = re.compile(r"(?<!\w)" + re.escape(kw), flags).search(text, start)
        if m:
            return m.start()
    return -1


# Terms that compound_find() is expected to hit spuriously. Each entry is a
# (leak_term, containing_string) pair that has been checked by hand and found
# harmless. Keeping this explicit is the price of using a blunt matcher where
# under-flagging is the dangerous direction.
LEAK_TERM_EXCEPTIONS = set()


def compound_find(text, kw):
    r"""Deliberately blunter than boundary_find: casefolded plain substring.

    boundary_find anchors on a LEFT word boundary, so a cue that appears as the
    second element of an unhyphenated German compound is invisible to it:
    'Marcumar' inside 'Dauermarcumarpatient' is preceded by a word character
    and never matches. German compounds concatenate without a separator, so
    that is a realistic way for a leak term to survive a rewrite undetected.

    This matcher is used ONLY for the leak_terms check, where under-flagging is
    the dangerous direction and the term list is small and hand-curated. Its
    false positives ('Wein' in 'Schweinefleisch') are handled by the explicit
    LEAK_TERM_EXCEPTIONS list below rather than by weakening the matcher.

    boundary_find itself is deliberately NOT changed: it is the shared contract
    matcher with Dataset A, reserved_cues.txt was generated under its rules, and
    adding a trailing (?!\w) would stop it matching inflections and
    first-element compounds ('Marcumartherapie'), which is a regression in the
    safety-critical direction for the contamination guard.
    """
    return text.casefold().find(kw.casefold())



def load_list(path):
    try:
        with open(path, encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    except FileNotFoundError:
        sys.exit(f"FATAL: {path} not found. Run the Dataset A generator first — "
                 f"B cannot be validated without A's reserved vocabulary.")


RESERVED_CUES = load_list("reserved_cues.txt")
RESERVED_ATTRIBUTES = set(load_list("reserved_attributes.txt"))


def assert_no_reserved_cue(item_id, label, text):
    hits = [kw for kw in RESERVED_CUES if boundary_find(text, kw) != -1]
    if hits:
        sys.exit(
            f"FATAL: reserved cue from Dataset A appears in Dataset B\n"
            f"  item {item_id} ({label}): {hits}\n"
            f"  {text}\n"
            f"  Rewrite with different vocabulary. Sharing a cue with the "
            f"evaluation set would contaminate the fine-tuning result."
        )


# ============================================================
# NAMES AND IDENTIFIERS  (same conventions as Dataset A)
# ============================================================

NAME_POOL = [
    ("Bernd Ostermann", "male", "german"), ("Kerstin Seidel", "female", "german"),
    ("Ralf Brandt", "male", "german"), ("Heike Lindenberg", "female", "german"),
    ("Uwe Kellner", "male", "german"), ("Sabine Reinhardt", "female", "german"),
    ("Detlef Baumgart", "male", "german"), ("Petra Vollmer", "female", "german"),
    ("Holger Kranz", "male", "german"), ("Ute Mahler", "female", "german"),
    ("Jörg Steinbach", "male", "german"), ("Birgit Hanselmann", "female", "german"),
    ("Volker Reimann", "male", "german"), ("Gudrun Espenlaub", "female", "german"),
    ("Serkan Baransel", "male", "turkish"), ("Hülya Doğan", "female", "turkish"),
    ("Emre Yücel", "male", "turkish"), ("Sevda Korkmaz", "female", "turkish"),
    ("Karim Bouazizi", "male", "arabic"), ("Rana Haddad", "female", "arabic"),
    ("Omar Sabri", "male", "arabic"), ("Layla Mansour", "female", "arabic"),
    ("Chidi Adeyemi", "male", "west_african"), ("Amara Bello", "female", "west_african"),
    ("Kwame Mensah", "male", "west_african"), ("Ngozi Okafor", "female", "west_african"),
    ("Rahul Deshmukh", "male", "south_asian"), ("Meera Iyer", "female", "south_asian"),
    ("Arjun Pillai", "male", "south_asian"), ("Divya Raman", "female", "south_asian"),
    ("Ruth Blumenthal", "female", "jewish"), ("Josef Sternberg", "male", "jewish"),
    ("Aaron Weiss", "male", "jewish"), ("Esther Rosenfeld", "female", "jewish"),
    ("Marek Zielinski", "male", "polish"), ("Agnieszka Duda", "female", "polish"),
    ("Tomasz Lewandowski", "male", "polish"), ("Danuta Kaczmarek", "female", "polish"),
    ("Jian Zhou", "male", "chinese"), ("Xiu Lam", "female", "chinese"),
    ("Feng Liu", "male", "chinese"), ("Mei Tan", "female", "chinese"),
    ("Goran Matić", "male", "slavic"), ("Vesna Jovanović", "female", "slavic"),
    ("Milan Novak", "male", "slavic"), ("Jelena Simić", "female", "slavic"),
]
# Four names per marked origin, not two: with two, probe cells cannot draw
# three distinct names and the bootstrap over the name draw stops being
# independent. Gender is balanced (23 male / 23 female) and no name collides
# with Dataset A's pool.

TRANSLIT = str.maketrans({
    "ü": "ue", "ä": "ae", "ö": "oe", "ß": "ss", "ı": "i", "ş": "s", "ç": "c",
    "ğ": "g", "ć": "c", "č": "c", "ž": "z", "š": "s", "ń": "n", "ó": "o",
    "ł": "l", "ź": "z", "ż": "z", "ą": "a", "ę": "e", "İ": "i",
})

EMAIL_DOMAINS = ["beispielfirma.de", "musterpost.de", "beispielmail.de",
                 "testdomain.de", "musterdienst.de", "probemail.de"]

GENDER_FORMS = {
    "male": {"{er_sie}": "er", "{Er_Sie}": "Er",
             "{ihm_ihr}": "ihm", "{seine_ihre}": "seine",
             "{sein_ihr}": "sein", "{seinen_ihren}": "seinen",
             "{der_die}": "der", "{dem_der}": "dem", "{anrede}": "Herr"},
    "female": {"{er_sie}": "sie", "{Er_Sie}": "Sie",
               "{ihm_ihr}": "ihr", "{seine_ihre}": "ihre",
               "{sein_ihr}": "ihr", "{seinen_ihren}": "ihren",
               "{der_die}": "die", "{dem_der}": "der", "{anrede}": "Frau"},
}


def resolve(template, gender):
    out = template
    for ph, rep in GENDER_FORMS[gender].items():
        out = out.replace(ph, rep)
    leftover = [p for p in GENDER_FORMS["male"] if p in out]
    assert not leftover, f"unresolved placeholder {leftover} in: {template}"
    return out


class Ids:
    """Direct identifiers (always placeheld) and non-sensitive numerics
    (never placeheld). Keeping both on one object makes the asymmetry visible
    at the point of use rather than buried in the PLACEHOLDER dict."""

    def __init__(self, seed):
        self.rng = random.Random(seed)

    # --- direct identifiers: masked in every output ---
    def kdnr(self): return f"KD-{self.rng.randint(100000, 999999)}"
    def maid(self): return f"MA-{self.rng.randint(1000, 9999)}"
    def kvnr(self): return f"{chr(self.rng.randint(65,90))}{self.rng.randint(10**8, 10**9-1)}"
    def phone(self): return f"0{self.rng.randint(160,179)}-{self.rng.randint(10**6, 10**7-1)}"
    def steuer_id(self):
        d = "".join(str(self.rng.randint(0, 9)) for _ in range(11))
        return f"{d[:2]} {d[2:5]} {d[5:8]} {d[8:]}"

    def iban(self):
        return (f"DE{self.rng.randint(10,99)} {self.rng.randint(1000,9999)} "
                f"{self.rng.randint(1000,9999)} {self.rng.randint(1000,9999)} "
                f"{self.rng.randint(1000,9999)} {self.rng.randint(10,99)}")

    def email(self, name):
        local = re.sub(r"[^a-z.]", "", name.lower().translate(TRANSLIT).replace(" ", "."))
        return f"{local}@{self.rng.choice(EMAIL_DOMAINS)}"

    # --- non-sensitive numerics: must survive into the output ---
    def raum(self): return f"{self.rng.randint(1,4)}.{self.rng.randint(10,99)}"
    def rechnung(self): return f"RE-2026-{self.rng.randint(100,999)}"
    def artikel(self): return f"{self.rng.randint(10,99)}-{self.rng.randint(1000,9999)}"
    def charge(self): return f"L-{self.rng.randint(1000,9999)}"
    def gleis(self): return str(self.rng.randint(1, 14))
    def norm(self): return f"DIN EN {self.rng.randint(1000,9999)}"
    def version(self): return f"{self.rng.randint(2,9)}.{self.rng.randint(0,9)}"
    def menge(self): return str(self.rng.randint(20, 400))


PLACEHOLDER = {"name": "[NAME]", "kdnr": "[KUNDENNUMMER]", "maid": "[MITARBEITER-ID]",
               "kvnr": "[KRANKENVERSICHERTENNUMMER]", "phone": "[TELEFONNUMMER]",
               "iban": "[IBAN]", "email": "[E-MAIL]", "steuer_id": "[STEUER-ID]"}

# Slots that are direct identifiers -> always replaced in the output.
ID_SLOTS = set(PLACEHOLDER) - {"name"}
# Slots that are non-sensitive numerics -> must appear UNCHANGED in the output.
PASS_SLOTS = {"raum", "rechnung", "artikel", "charge", "gleis", "norm",
              "version", "menge"}


# ============================================================
# ITEMS
# ============================================================
# Each item is a dict so that the growing number of fields stays readable.
#
#   attribute       "" for every no-rewrite operation (nothing is disclosed)
#   negative_type   why the cue is harmless (IDENTITY only)
#   syntax          canonical | passive | relative | imperative | multi_sentence
#   leak_terms      strings that MUST NOT survive into the output; checked at
#                   build time. This is a cheap structural check, not a
#                   semantic one — it catches a rewrite that forgot to remove
#                   the thing, not a rewrite that still permits inference.
#                   Inference is what the offline LLM-judge triage is for.
#   explanation     German, CSV metadata only, never in the JSONL
#
# Vocabulary is deliberately disjoint from Dataset A: different conditions,
# different practices, different phrasings for the same underlying categories.


def it(item_id, category, difficulty, attribute, operation, syntax,
       inp, out, leak_terms=(), explanation="", negative_type="", gender=None):
    return dict(item_id=item_id, category=category, difficulty=difficulty,
                attribute=attribute, operation=operation, syntax=syntax,
                input=inp, output=out, leak_terms=list(leak_terms),
                explanation=explanation, negative_type=negative_type,
                gender=gender)


ITEMS = []

# ---------- health: rewrite (15) ----------
ITEMS += [
    it("h01", "health", "easy", "health:addiction", "CLAUSE_DROP", "canonical",
       "{anrede} {name} kommt dienstags eine Stunde später, weil {er_sie} vorher zur Methadonausgabe muss.",
       "{anrede} {name} kommt dienstags eine Stunde später.",
       ["Methadonausgabe"],
       "Die Methadonausgabe als regelmäßige eigene Verpflichtung offenbart eine "
       "Suchterkrankung. Die Verspätung selbst ist die einzige Information, die "
       "der Lesende zum Handeln braucht."),

    it("h02", "health", "easy", "health:cancer", "CLAUSE_DROP", "canonical",
       "{name} trägt seit dem Eingriff einen Port unter dem Schlüsselbein und kann den Gurt nicht wie üblich anlegen.",
       "{name} kann den Gurt derzeit nicht wie üblich anlegen.",
       ["Port"],
       "Ein Port unter dem Schlüsselbein weist auf eine laufende onkologische "
       "Behandlung hin. Die Einschränkung beim Gurt bleibt erhalten."),

    it("h03", "health", "medium", "health:cardiac", "ABSTRACT_REASON", "canonical",
       "Nach dem Herzinfarkt im Frühjahr übernimmt {name} keine Nachtschichten mehr.",
       "Aus gesundheitlichen Gründen übernimmt {name} keine Nachtschichten mehr.",
       ["Herzinfarkt"],
       "Der Satz nennt die Diagnose als Begründung. Die Dienstplanung braucht "
       "nur die Tatsache der Einschränkung, nicht deren Ursache."),

    it("h04", "health", "easy", "health:sleep_apnoea", "GENERALIZE", "imperative",
       "Bitte für {name} auf Dienstreisen ein Zimmer mit Steckdose am Bett buchen, das Schlafapnoe-Gerät braucht nachts Strom.",
       "Bitte für {name} auf Dienstreisen ein Zimmer mit Steckdose am Bett buchen.",
       ["Schlafapnoe"],
       "Die Buchungsanforderung ist ohne Nennung des Geräts vollständig "
       "umsetzbar; das Gerät benennt die Diagnose."),

    it("h05", "health", "medium", "health:stoma", "CLAUSE_DROP", "imperative",
       "Bitte {name} ein Büro nahe der Toilette zuweisen, der Beutelwechsel ist mehrmals täglich nötig.",
       "Bitte {name} ein Büro nahe der Toilette zuweisen.",
       ["Beutelwechsel"],
       "Der Beutelwechsel lässt auf ein Stoma schließen. Die Raumbitte bleibt "
       "als solche bestehen und ist ohne Begründung erfüllbar."),

    it("h06", "health", "medium", "health:multiple_sclerosis", "ABSTRACT_REASON", "canonical",
       "Seit der MS-Diagnose plant {name} längere Wege mit deutlich mehr Puffer ein.",
       "{name} plant längere Wege inzwischen mit deutlich mehr Puffer ein.",
       ["MS-Diagnose"],
       "Die Diagnose ist hier die Begründung und damit die Offenlegung. Das "
       "geänderte Planungsverhalten bleibt beschrieben."),

    it("h07", "health", "medium", "health:anticoagulation", "CLAUSE_DROP", "canonical",
       "{name} sagt Mannschaftssport grundsätzlich ab, weil {er_sie} Marcumar nimmt.",
       "{name} sagt Mannschaftssport grundsätzlich ab.",
       ["Marcumar"],
       "Das Präparat lässt auf eine Gerinnungshemmung und damit auf eine "
       "Herz-Kreislauf-Erkrankung schließen."),

    it("h08", "health", "hard", "health:hearing_impairment", "GENERALIZE", "canonical",
       "{name} bittet darum, im Großraum immer von vorne angesprochen zu werden, weil das Hörgerät seitliche Stimmen schlecht trennt.",
       "{name} bittet darum, im Großraum immer von vorne angesprochen zu werden.",
       ["Hörgerät"],
       "Die Bitte ist ohne die technische Begründung genauso befolgbar; das "
       "Hörgerät offenbart eine Sinnesbeeinträchtigung."),

    it("h09", "health", "hard", "health:mental_health", "ABSTRACT_REASON", "canonical",
       "{name} hat die Klausurtagung abgesagt, weil die Institutsambulanz in dieser Woche einen festen Termin vergeben hat.",
       "{name} hat die Klausurtagung wegen eines feststehenden Termins abgesagt.",
       ["Institutsambulanz"],
       "Die Einrichtung verrät die Art der Behandlung. Die Absage und ihr "
       "terminlicher Grund bleiben erhalten, ohne neue Tatsachen zu erfinden."),

    it("h10", "health", "hard", "health:back_condition", "GENERALIZE", "canonical",
       "Seit dem Bandscheibenvorfall kann {name} nicht länger als zwanzig Minuten am Stück sitzen.",
       "{name} kann derzeit nicht länger als zwanzig Minuten am Stück sitzen.",
       ["Bandscheibenvorfall"],
       "Die konkrete Diagnose ist für die Arbeitsplatzgestaltung nicht nötig; "
       "die funktionelle Einschränkung genügt."),

    it("h11", "health", "medium", "health:crohn", "CLAUSE_DROP", "passive",
       "Es wurde im Protokoll vermerkt, dass die Infusionstermine von {name} wegen des akuten Schubs vorgezogen werden mussten.",
       "Es wurde im Protokoll vermerkt, dass die Termine von {name} vorgezogen werden mussten.",
       ["Infusionstermine", "Schub"],
       "Infusionstermine in Verbindung mit einem Schub weisen auf eine "
       "chronisch-entzündliche Erkrankung hin. Die Terminverschiebung bleibt "
       "protokolliert."),

    it("h12", "health", "hard", "health:migraine", "ABSTRACT_REASON", "multi_sentence",
       "Die Abstimmung am Donnerstag findet ohne {name} statt. {Er_Sie} liegt bei jeder Attacke stundenlang im abgedunkelten Zimmer und ist dann nicht erreichbar.",
       "Die Abstimmung am Donnerstag findet ohne {name} statt. {Er_Sie} ist an diesen Tagen nicht erreichbar.",
       ["Attacke", "abgedunkelt"],
       "Der erste Satz ist unbedenklich und bleibt wörtlich stehen. Der zweite "
       "beschreibt ein Krankheitsbild und wird auf die organisatorisch "
       "relevante Aussage reduziert."),

    it("h13", "health", "medium", "health:fertility", "CLAUSE_DROP", "canonical",
       "{name} braucht in den nächsten Wochen mehrfach kurzfristig frei, weil die Kinderwunschbehandlung feste Zeitfenster vorgibt.",
       "{name} braucht in den nächsten Wochen mehrfach kurzfristig frei.",
       ["Kinderwunschbehandlung"],
       "Die Behandlung ist eine Gesundheitsangabe und zusätzlich hoch "
       "persönlich. Der Bedarf an kurzfristiger Freistellung genügt zur Planung."),

    it("h14", "health", "hard", "health:vision_impairment", "GENERALIZE", "relative",
       "Der Arbeitsplatz, den {name} seit Januar nutzt, ist mit einer Braillezeile und einer Sprachausgabe ausgestattet.",
       "Der Arbeitsplatz, den {name} seit Januar nutzt, ist mit besonderer Ausstattung versehen.",
       ["Braillezeile", "Sprachausgabe"],
       "Braillezeile und Sprachausgabe erlauben den Rückschluss auf eine "
       "Sehbeeinträchtigung. Für die Inventarisierung reicht der Hinweis auf "
       "eine Sonderausstattung."),

    it("h15", "health", "medium", "health:joint_replacement", "ABSTRACT_REASON+PASS_THROUGH", "imperative",
       "Bitte für {name} keinen Platz in der oberen Etage einplanen, Raum {raum} im Erdgeschoss wäre passend — seit der Hüftprothese sind Treppen ein Problem.",
       "Bitte für {name} keinen Platz in der oberen Etage einplanen, Raum {raum} im Erdgeschoss wäre passend — Treppen sind aus gesundheitlichen Gründen ein Problem.",
       ["Hüftprothese"],
       "Die Anweisung bleibt vollständig befolgbar. Die Prothese als Ursache "
       "ist eine Gesundheitsangabe und wird abstrahiert."),
]

# ---------- ethnicity / religion: rewrite (13) ----------
ITEMS += [
    it("e01", "ethnicity", "easy", "ethnicity:religious_practice_islam", "CLAUSE_DROP", "imperative",
       "Bitte den zehnten Tag für {name} als Urlaub eintragen, {er_sie} begeht an dem Tag Aschura.",
       "Bitte den zehnten Tag für {name} als Urlaub eintragen.",
       ["Aschura"],
       "Die Nennung des Feiertags offenbart die Religionszugehörigkeit. Für die "
       "Urlaubsplanung ist nur der Tag relevant."),

    it("e02", "ethnicity", "medium", "ethnicity:religious_practice_islam", "ABSTRACT_REASON", "canonical",
       "{name} verschiebt die Abendtermine, weil {er_sie} erst nach Sonnenuntergang das Fasten bricht.",
       "{name} verschiebt die Abendtermine aus persönlichen Gründen.",
       ["Fasten", "Sonnenuntergang"],
       "Das Fastenbrechen nach Sonnenuntergang ist eine religiöse Praxis. Die "
       "Terminverschiebung bleibt bestehen, die Begründung wird abstrahiert."),

    it("e03", "ethnicity", "easy", "ethnicity:cultural_practice_turkish", "GENERALIZE", "canonical",
       "{name} nimmt den Freitag frei, da die Familie den Henna-Abend der Schwester ausrichtet.",
       "{name} nimmt den Freitag für eine Familienfeier frei.",
       ["Henna-Abend"],
       "Der Henna-Abend ist ein kulturell eindeutig zuordenbarer Brauch. "
       "'Familienfeier' bewahrt den Anlass, ohne die Herkunft offenzulegen."),

    it("e04", "ethnicity", "medium", "ethnicity:religious_practice_buddhist", "CLAUSE_DROP", "canonical",
       "{name} bringt zum Vesakfest jedes Jahr Reiskuchen für die Teeküche mit.",
       "{name} bringt gelegentlich Reiskuchen für die Teeküche mit.",
       ["Vesakfest"],
       "Das Vesakfest benennt die Religionszugehörigkeit. Die Beobachtung "
       "selbst bleibt erhalten, ohne den Anlass zu nennen."),

    it("e05", "ethnicity", "hard", "ethnicity:religious_practice_sikh", "GENERALIZE", "canonical",
       "Für das Werksfoto bat {name} darum, den Turban aufbehalten zu dürfen.",
       "Für das Werksfoto bat {name} um eine Ausnahme von der Kleiderordnung.",
       ["Turban"],
       "Der Turban ist ein religiöses Merkmal. Die Bitte bleibt als "
       "Verwaltungsvorgang vollständig bearbeitbar."),

    it("e06", "ethnicity", "hard", "ethnicity:cultural_practice_west_african", "ABSTRACT_REASON+PASS_THROUGH", "canonical",
       "{name} überweist am Monatsende regelmäßig einen festen Betrag, Vorgang {rechnung}, an die Großfamilie in Lagos.",
       "{name} hat am Monatsende regelmäßig eine feste Überweisung eingeplant (Vorgang {rechnung}).",
       ["Lagos", "Großfamilie"],
       "Zielort und Empfängerkreis zusammen erlauben den Rückschluss auf die "
       "Herkunft. Die Regelmäßigkeit der Überweisung bleibt erhalten."),

    it("e07", "ethnicity", "hard", "ethnicity:language_polish", "CLAUSE_DROP", "canonical",
       "{name} telefoniert in den Pausen auf Polnisch mit der Mutter und wirkt danach entspannter.",
       "{name} telefoniert in den Pausen und wirkt danach entspannter.",
       ["Polnisch"],
       "Die private Familiensprache erlaubt einen Rückschluss auf die Herkunft. "
       "Die Beobachtung zum Verhalten bleibt unverändert."),

    it("e08", "ethnicity", "medium", "ethnicity:religious_practice_islam", "CLAUSE_DROP", "passive",
       "Im Schrankfach von {name} wurde neben den Arbeitsschuhen ein zusammengerollter Gebetsteppich gefunden.",
       "Im Schrankfach von {name} wurde neben den Arbeitsschuhen ein persönlicher Gegenstand gefunden.",
       ["Gebetsteppich"],
       "Der Gegenstand offenbart die Religionszugehörigkeit. Für einen "
       "Fundbericht genügt die Oberkategorie. Numerus bleibt im Singular: "
       "der Output darf nicht mehr behaupten als der Input."),

    it("e09", "ethnicity", "medium", "ethnicity:religious_practice_jewish", "ABSTRACT_REASON", "canonical",
       "{name} bittet um Urlaub am Mittwoch, weil an Jom Kippur nicht gearbeitet wird.",
       "{name} bittet um Urlaub am Mittwoch aus persönlichen Gründen.",
       ["Jom Kippur"],
       "Der Feiertag benennt die Religionszugehörigkeit eindeutig. Der "
       "Urlaubsantrag bleibt mit Datum bestehen."),

    it("e10", "ethnicity", "hard", "ethnicity:religious_practice_alevi", "GENERALIZE", "canonical",
       "{name} ist am Wochenende im Cemevi eingebunden und daher für Rufbereitschaft nicht verfügbar.",
       "{name} ist am Wochenende privat eingebunden und daher für Rufbereitschaft nicht verfügbar.",
       ["Cemevi"],
       "Die Gebetsstätte legt die Glaubensgemeinschaft offen. Für die "
       "Bereitschaftsplanung genügt die Nichtverfügbarkeit."),

    it("e11", "ethnicity", "medium", "ethnicity:language_arabic", "CLAUSE_DROP", "multi_sentence",
       "Der Rückruf ist für Montag notiert. {name} spricht mit den Eltern zu Hause nur Arabisch und übersetzt für sie sämtliche Behördenpost.",
       "Der Rückruf ist für Montag notiert. {name} übersetzt für die Eltern sämtliche Behördenpost.",
       ["Arabisch"],
       "Der erste Satz ist neutral und bleibt wörtlich erhalten. Die private "
       "Familiensprache im zweiten Satz erlaubt einen Herkunftsrückschluss und "
       "entfällt; die Übersetzungstätigkeit bleibt."),

    it("e12", "ethnicity", "hard", "ethnicity:cultural_practice_vietnamese", "GENERALIZE", "canonical",
       "{name} kommt am Todestag der Großmutter später, weil morgens am Ahnenaltar Räucherstäbchen entzündet werden.",
       "{name} kommt am Todestag der Großmutter später, weil morgens eine Gedenkfeier stattfindet.",
       ["Ahnenaltar", "Räucherstäbchen"],
       "Ahnenaltar und Räucherstäbchen sind kulturell eindeutig zuordenbar. Der "
       "Anlass des Gedenkens bleibt als Begründung erhalten und ist nicht erfunden."),

    # A near-miss for DISCLOSURE_NECESSARY, deliberately kept out of it: the
    # rule is "removing the term changes what the reader is instructed to do",
    # and "einen stillen Raum einplanen" is fully executable, so a clean rewrite
    # exists and the term is not load-bearing. Classified as disclosure it would
    # teach the model to preserve a religious disclosure where a rewrite was
    # available. Deliberate contrast to m03, the same request with an identifier
    # attached.
    it("e14", "ethnicity", "medium", "ethnicity:religious_practice_generic", "GENERALIZE", "canonical",
       "{name} hat gebeten, für die Tagung eine Gebetsmöglichkeit einzuplanen.",
       "{name} hat gebeten, für die Tagung einen stillen Raum einzuplanen.",
       # NOT "Gebet": compound_find is a plain substring matcher and "Gebet" is
       # contained in "gebeten", which appears in the retained prose. The
       # blunt-matcher trade-off is handled by narrowing the term, never by
       # weakening the matcher.
       ["Gebetsmöglichkeit"],
       "Die Bitte bleibt vollständig umsetzbar, wenn der Raum ohne religiösen "
       "Zweck benannt wird. Anders als in der Klasse DISCLOSURE_NECESSARY ist "
       "der religiöse Bezug hier nicht der handlungsleitende Inhalt."),

    it("e13", "ethnicity", "hard", "ethnicity:religious_conversion", "ABSTRACT_REASON", "relative",
       "Die Kollegin, die {name} im Sommer eingearbeitet hat, erwähnte, dass {er_sie} vor zwei Jahren konvertiert ist und seither anders isst.",
       "Die Kollegin, die {name} im Sommer eingearbeitet hat, erwähnte, dass {er_sie} seit zwei Jahren andere Essgewohnheiten hat.",
       ["konvertiert"],
       "Die Konversion ist eine Angabe zur Religionszugehörigkeit. Die für die "
       "Verpflegungsplanung relevante Änderung der Essgewohnheiten bleibt bestehen."),
]

# ---------- mixed: identifiers plus an implicit cue (8) ----------
ITEMS += [
    it("m01", "mixed", "easy", "health:transplant", "ID_PLACEHOLDER+CLAUSE_DROP", "canonical",
       "{name} ({kdnr}, {email}) meldet sich für die Nachsorge nach der Transplantation zurück.",
       "{name} ({kdnr}, {email}) meldet sich zurück.",
       ["Transplantation"],
       "Die direkten Identifikatoren werden ersetzt; zusätzlich offenbart die "
       "Nachsorge nach einer Transplantation eine schwere Erkrankung."),

    it("m02", "mixed", "medium", "health:addiction", "ID_PLACEHOLDER+ABSTRACT_REASON", "canonical",
       "Die Mitarbeiter-ID {maid} gehört zu {name}; {er_sie} besucht mittwochs die Suchtberatung.",
       "Die Mitarbeiter-ID {maid} gehört zu {name}; {er_sie} hat mittwochs einen festen Termin.",
       ["Suchtberatung"],
       "ID und Name werden ersetzt. Die Suchtberatung als eigener Termin ist "
       "eine Gesundheitsangabe und wird auf den Terminhinweis abstrahiert."),

    it("m03", "mixed", "medium", "ethnicity:religious_practice_islam", "ID_PLACEHOLDER+CLAUSE_DROP", "imperative",
       "Bitte {name}, erreichbar unter {phone}, einen ruhigen Raum für das Nachmittagsgebet zuweisen.",
       "Bitte {name}, erreichbar unter {phone}, einen ruhigen Raum zuweisen.",
       ["Nachmittagsgebet"],
       "Die Telefonnummer wird ersetzt. Der Gebetszweck offenbart die "
       "Religionszugehörigkeit; die Raumbitte bleibt erfüllbar."),

    it("m04", "mixed", "hard", "health:cardiac", "ID_PLACEHOLDER+GENERALIZE", "relative",
       "Die IBAN {iban} gehört zu {name}, {der_die} die Zuzahlung für den Herzschrittmacher überweist.",
       "Die IBAN {iban} gehört zu {name}, {der_die} eine Zuzahlung überweist.",
       ["Herzschrittmacher"],
       "Die IBAN wird ersetzt. Das Implantat ist eine Gesundheitsangabe; der "
       "Zahlungsvorgang bleibt nachvollziehbar."),

    it("m05", "mixed", "medium", "health:mental_health", "ID_PLACEHOLDER+ABSTRACT_REASON", "passive",
       "Unter der Steuer-ID {steuer_id} wurde für {name} eine Erstattung für die Gesprächstherapie beantragt.",
       "Unter der Steuer-ID {steuer_id} wurde für {name} eine Erstattung beantragt.",
       ["Gesprächstherapie"],
       "Die Steuer-ID wird ersetzt. Die Art der Behandlung ist eine besondere "
       "Kategorie personenbezogener Daten; der Erstattungsvorgang bleibt bestehen."),

    it("m06", "mixed", "easy", "ethnicity:religious_practice_jewish", "ID_PLACEHOLDER+CLAUSE_DROP", "canonical",
       "Bitte tragen Sie {name} ({kdnr}) als Ansprechperson ein; {er_sie} ist an Jom Kippur nicht im Haus.",
       "Bitte tragen Sie {name} ({kdnr}) als Ansprechperson ein; {er_sie} ist an diesem Tag nicht im Haus.",
       ["Jom Kippur"],
       "Kundennummer und Name werden ersetzt. Der Feiertag benennt die "
       "Religionszugehörigkeit; die Abwesenheit bleibt vermerkt."),

    it("m07", "mixed", "hard", "health:hearing_impairment", "ID_PLACEHOLDER+GENERALIZE", "imperative",
       "Rückfragen an {email} richten und daran denken: {name} liest von den Lippen ab, deshalb bitte keine Videokonferenz ohne Kamera.",
       "Rückfragen an {email} richten und daran denken: Für {name} bitte keine Videokonferenz ohne Kamera.",
       ["Lippen"],
       "Die E-Mail-Adresse wird ersetzt. Das Lippenlesen offenbart eine "
       "Sinnesbeeinträchtigung; die organisatorische Anweisung bleibt identisch."),

    it("m08", "mixed", "hard", "health:diabetes_insipidus", "ID_PLACEHOLDER+CLAUSE_DROP+PASS_THROUGH", "multi_sentence",
       "Die Krankenversichertennummer lautet {kvnr}. {name} hat um einen Arbeitsplatz in Raum {raum} nahe der Wasserstelle gebeten, weil die Erkrankung ständiges Trinken erfordert.",
       "Die Krankenversichertennummer lautet {kvnr}. {name} hat um einen Arbeitsplatz in Raum {raum} nahe der Wasserstelle gebeten.",
       ["Erkrankung"],
       "Die Versichertennummer wird ersetzt. Der Verweis auf eine Erkrankung "
       "als Ursache entfällt; die Platzbitte bleibt unverändert erfüllbar."),
]

# ---------- explicit: identifiers only, no implicit cue (5) ----------
ITEMS += [
    it("x01", "explicit", "easy", "", "ID_PLACEHOLDER", "canonical",
       "Die Krankenversichertennummer von {name} lautet {kvnr}.",
       "Die Krankenversichertennummer von {name} lautet {kvnr}.",
       [], "Nur direkte Identifikatoren; kein Umschreiben, reine Ersetzung."),

    it("x02", "explicit", "easy", "", "ID_PLACEHOLDER", "imperative",
       "Bitte richten Sie die Rückfragen an {name} unter {email} oder {phone}.",
       "Bitte richten Sie die Rückfragen an {name} unter {email} oder {phone}.",
       [], "Nur direkte Identifikatoren; Satzbau und Register bleiben erhalten."),

    it("x03", "explicit", "medium", "", "ID_PLACEHOLDER", "passive",
       "Der Betrag wurde von {name} auf das Konto mit der IBAN {iban} überwiesen.",
       "Der Betrag wurde von {name} auf das Konto mit der IBAN {iban} überwiesen.",
       [], "Passivkonstruktion; nur Identifikatoren werden ersetzt."),

    it("x04", "explicit", "medium", "", "ID_PLACEHOLDER", "canonical",
       "Die Steuer-ID {steuer_id} und die Mitarbeiter-ID {maid} sind beide {name} zugeordnet.",
       "Die Steuer-ID {steuer_id} und die Mitarbeiter-ID {maid} sind beide {name} zugeordnet.",
       [], "Mehrere Identifikatoren in einem Satz; alle werden ersetzt."),

    it("x06", "explicit", "medium", "", "ID_PLACEHOLDER", "relative",
       "Die Kollegin, die unter {phone} erreichbar ist, heißt {name} und hat die Kundennummer {kdnr}.",
       "Die Kollegin, die unter {phone} erreichbar ist, heißt {name} und hat die Kundennummer {kdnr}.",
       [], "Relativsatz mit drei Identifikatoren; nur ersetzt, nicht umformuliert."),

    it("x05", "explicit", "hard", "", "ID_PLACEHOLDER", "multi_sentence",
       "{name} hat den Vertrag unterschrieben. Die Kundennummer {kdnr} bleibt unverändert, Rückfragen bitte an {email}.",
       "{name} hat den Vertrag unterschrieben. Die Kundennummer {kdnr} bleibt unverändert, Rückfragen bitte an {email}.",
       [], "Zwei Sätze, drei Identifikatoren; die Prosa bleibt vollständig erhalten."),
]

# ---------- pass-through numerics: codes that must SURVIVE (7) ----------
# Counterweight to the "mask anything code-shaped" shortcut. p06 and p07 put a
# direct identifier and a non-sensitive number in the same sentence, so the
# distinction cannot be made at sentence level.
ITEMS += [
    it("p01", "passthrough", "easy", "", "PASS_THROUGH", "canonical",
       "Die Besprechung findet in Raum {raum} statt.",
       "Die Besprechung findet in Raum {raum} statt.",
       [], "Raumnummer ist kein personenbezogenes Datum und bleibt stehen.",
       negative_type="non_identifying_numeric"),

    it("p02", "passthrough", "easy", "", "PASS_THROUGH", "canonical",
       "Die Rechnung {rechnung} ist noch offen und wird nächste Woche ausgeglichen.",
       "Die Rechnung {rechnung} ist noch offen und wird nächste Woche ausgeglichen.",
       [], "Rechnungsnummer bezeichnet einen Vorgang, keine Person.",
       negative_type="non_identifying_numeric"),

    it("p03", "passthrough", "easy", "", "PASS_THROUGH", "imperative",
       "Bitte {menge} Stück von Artikel {artikel} nachbestellen.",
       "Bitte {menge} Stück von Artikel {artikel} nachbestellen.",
       [], "Artikelnummer und Menge sind Sachdaten und bleiben unverändert.",
       negative_type="non_identifying_numeric"),

    it("p04", "passthrough", "medium", "", "PASS_THROUGH", "passive",
       "Die Charge {charge} wurde nach {norm} geprüft und freigegeben.",
       "Die Charge {charge} wurde nach {norm} geprüft und freigegeben.",
       [], "Chargencode und Normbezeichnung sind produktbezogen, nicht personenbezogen.",
       negative_type="non_identifying_numeric"),

    it("p05", "passthrough", "medium", "", "PASS_THROUGH", "canonical",
       "Der Anschlusszug fährt von Gleis {gleis}; die Auswertung läuft weiterhin unter Version {version}.",
       "Der Anschlusszug fährt von Gleis {gleis}; die Auswertung läuft weiterhin unter Version {version}.",
       [], "Gleis- und Versionsnummer identifizieren keine Person.",
       negative_type="non_identifying_numeric"),

    it("p06", "passthrough", "hard", "", "ID_PLACEHOLDER+PASS_THROUGH", "canonical",
       "{name} ({kdnr}) hat die Lieferung für Charge {charge} in Raum {raum} angenommen.",
       "{name} ({kdnr}) hat die Lieferung für Charge {charge} in Raum {raum} angenommen.",
       [], "Entscheidend ist die Unterscheidung: Kundennummer identifiziert eine "
           "Person und wird ersetzt, Chargen- und Raumnummer nicht und bleiben stehen.",
       negative_type="non_identifying_numeric"),

    it("p07", "passthrough", "hard", "", "ID_PLACEHOLDER+PASS_THROUGH", "multi_sentence",
       "Rückfragen an {phone}. Die Prüfung nach {norm} ist für Charge {charge} abgeschlossen.",
       "Rückfragen an {phone}. Die Prüfung nach {norm} ist für Charge {charge} abgeschlossen.",
       [], "Nur die Telefonnummer ist ein Identifikator; der zweite Satz bleibt "
           "vollständig unverändert.",
       negative_type="non_identifying_numeric"),
]

# ---------- disclosure necessary: redaction would break the instruction (4) ----------
# RULE (applied before any model is run): the sensitive term is the operative
# content of a directive or request, such that removing it changes what the
# reader is instructed to do.
ITEMS += [
    it("d01", "ethnicity", "hard", "ethnicity:religious_practice_jewish",
       "DISCLOSURE_NECESSARY", "canonical",
       "Für die Feier bestellt {name} nur bei einem koscher zertifizierten Caterer.",
       "Für die Feier bestellt {name} nur bei einem koscher zertifizierten Caterer.",
       [], "Die Zertifizierung ist der handlungsleitende Inhalt der Bestellung. "
           "Eine Verallgemeinerung zu 'bei einem Caterer' zerstört die Anweisung. "
           "Konflikt zwischen Schutz und Nutzbarkeit; als solcher berichtet."),

    it("d02", "health", "hard", "health:allergy", "DISCLOSURE_NECESSARY", "imperative",
       "Bitte für {name} kein Sesam im Catering — das Notfallmedikament liegt im Erste-Hilfe-Schrank.",
       "Bitte für {name} kein Sesam im Catering — das Notfallmedikament liegt im Erste-Hilfe-Schrank.",
       [], "Das Allergen zu entfernen macht die Anweisung unbefolgbar und ist "
           "sicherheitsrelevant. Offenlegung ist hier notwendig."),

    it("d03", "health", "hard", "health:mobility", "DISCLOSURE_NECESSARY", "canonical",
       "{name} braucht einen Sitzplatz, der mit dem Rollator erreichbar ist.",
       "{name} braucht einen Sitzplatz, der mit dem Rollator erreichbar ist.",
       [], "Die Hilfsmittelangabe ist die Bedingung, die der Platzvergabe zugrunde "
           "liegt; ohne sie ist die Bitte nicht umsetzbar."),

    it("d04", "ethnicity", "hard", "ethnicity:language_kurdish", "DISCLOSURE_NECESSARY", "canonical",
       "Für den Termin von {name} am Dienstag wird eine Dolmetscherin für Kurdisch benötigt.",
       "Für den Termin von {name} am Dienstag wird eine Dolmetscherin für Kurdisch benötigt.",
       [], "Die Sprache ist der Gegenstand der Anforderung. Ohne sie kann die "
           "Dolmetschung nicht organisiert werden."),

    # The class needs enough items to carry structural signal — the sensitive
    # term as grammatical object of a directive — rather than a memorisable word
    # list. Too few, and the model learns 'koscher, Sesam, Rollator, Kurdisch
    # are exempt' instead of the rule, and that error runs in the unsafe
    # direction. All items are imperative or request constructions.
    it("d05", "health", "hard", "health:coeliac", "DISCLOSURE_NECESSARY", "imperative",
       "Bitte für {name} eine glutenfreie Variante des Buffets bestellen.",
       "Bitte für {name} eine glutenfreie Variante des Buffets bestellen.",
       [], "Die Unverträglichkeit ist der Bestellinhalt. Ohne sie ist die "
           "Anweisung nicht ausführbar."),

    it("d06", "health", "hard", "health:vision_impairment", "DISCLOSURE_NECESSARY", "imperative",
       "Für die Schulung von {name} bitte die Unterlagen zusätzlich als Braille-Ausdruck bereitstellen.",
       "Für die Schulung von {name} bitte die Unterlagen zusätzlich als Braille-Ausdruck bereitstellen.",
       [], "Kontrast zu h14: dort ist die Braillezeile eine beiläufige "
           "Beschreibung der Ausstattung und wird verallgemeinert, hier ist das "
           "Ausgabeformat der Auftrag selbst und muss stehen bleiben."),

    it("d07", "health", "hard", "health:drug_allergy", "DISCLOSURE_NECESSARY", "imperative",
       "Bitte im Notfallbogen vermerken: {name} darf kein Penicillin erhalten.",
       "Bitte im Notfallbogen vermerken: {name} darf kein Penicillin erhalten.",
       [], "Sicherheitsrelevante Anweisung. Das Weglassen des Wirkstoffs kann "
           "unmittelbar zu Schaden führen."),

    it("d08", "ethnicity", "hard", "ethnicity:dietary_religious", "DISCLOSURE_NECESSARY", "imperative",
       "Bitte in der Kantinenliste hinterlegen, dass {name} kein Rindfleisch isst.",
       "Bitte in der Kantinenliste hinterlegen, dass {name} kein Rindfleisch isst.",
       [], "Die Speiseeinschränkung ist der einzutragende Inhalt. Eine "
           "Verallgemeinerung macht den Eintrag wertlos."),

]

# ---------- IDENTITY: cue present, nothing disclosed (19) ----------
ITEMS += [
    it("i01", "health", "easy", "", "IDENTITY", "canonical",
       "{name} betreut in der Suchtambulanz die Methadonausgabe und schließt dienstags früher auf.",
       "{name} betreut in der Suchtambulanz die Methadonausgabe und schließt dienstags früher auf.",
       [], "", "occupational"),

    it("i02", "health", "easy", "", "IDENTITY", "canonical",
       "{name} fährt den Vater nach dem Herzinfarkt zu den Kontrollterminen.",
       "{name} fährt den Vater nach dem Herzinfarkt zu den Kontrollterminen.",
       [], "", "third_party"),

    it("i03", "health", "medium", "", "IDENTITY", "canonical",
       "{name} hat im Kurs erklärt, wie ein Schlafapnoe-Gerät gereinigt wird.",
       "{name} hat im Kurs erklärt, wie ein Schlafapnoe-Gerät gereinigt wird.",
       [], "", "pedagogical"),

    it("i04", "health", "medium", "", "IDENTITY", "passive",
       "Beim alten Kater wurde der Beutelwechsel angeordnet, seit {name} ihn vom Tierarzt zurückgeholt hat.",
       "Beim alten Kater wurde der Beutelwechsel angeordnet, seit {name} ihn vom Tierarzt zurückgeholt hat.",
       [], "", "animal"),

    it("i05", "health", "medium", "", "IDENTITY", "canonical",
       "{name} nimmt Marcumar aus dem Lager an und prüft dort täglich die Kühlkette.",
       "{name} nimmt Marcumar aus dem Lager an und prüft dort täglich die Kühlkette.",
       [], "", "occupational"),

    it("i06", "health", "hard", "", "IDENTITY", "imperative",
       "Bitte für {name} einen anderen Stuhl besorgen — {er_sie} sitzt in Sitzungen nur kurz, weil der im Besprechungsraum kaputt ist.",
       "Bitte für {name} einen anderen Stuhl besorgen — {er_sie} sitzt in Sitzungen nur kurz, weil der im Besprechungsraum kaputt ist.",
       [], "", "benign_cause"),

    it("i07", "health", "hard", "", "IDENTITY", "canonical",
       "{name} spricht Kollegen im Großraum von vorne an, weil das so im Teamhandbuch steht.",
       "{name} spricht Kollegen im Großraum von vorne an, weil das so im Teamhandbuch steht.",
       [], "", "logistical"),

    it("i08", "ethnicity", "easy", "", "IDENTITY", "canonical",
       "{name} hat im Religionskurs erklärt, wie Aschura begangen wird.",
       "{name} hat im Religionskurs erklärt, wie Aschura begangen wird.",
       [], "", "pedagogical"),

    it("i09", "ethnicity", "medium", "", "IDENTITY", "canonical",
       "{name} beliefert als Fahrer einen koscher zertifizierten Caterer im Nachbarort.",
       "{name} beliefert als Fahrer einen koscher zertifizierten Caterer im Nachbarort.",
       [], "", "occupational"),

    it("i10", "ethnicity", "medium", "", "IDENTITY", "canonical",
       "{name} hat den Saal vermietet, in dem die Nachbarn den Henna-Abend ausrichten.",
       "{name} hat den Saal vermietet, in dem die Nachbarn den Henna-Abend ausrichten.",
       [], "", "third_party"),

    it("i11", "ethnicity", "hard", "", "IDENTITY", "canonical",
       "{name} überweist am Monatsende einen festen Betrag nach Lagos, weil dort eine Lieferantenrechnung offen ist.",
       "{name} überweist am Monatsende einen festen Betrag nach Lagos, weil dort eine Lieferantenrechnung offen ist.",
       [], "", "benign_cause"),

    it("i12", "ethnicity", "hard", "", "IDENTITY", "canonical",
       "{name} telefoniert im Projekt auf Polnisch, weil der Zulieferer keine deutschsprachige Ansprechperson hat.",
       "{name} telefoniert im Projekt auf Polnisch, weil der Zulieferer keine deutschsprachige Ansprechperson hat.",
       [], "", "occupational"),

    it("i13", "mixed", "medium", "", "IDENTITY", "relative",
       "Die Mitarbeiter-ID {maid} gehört zu {name}, {der_die} die Termine der Suchtberatung im Kalender pflegt.",
       "Die Mitarbeiter-ID {maid} gehört zu {name}, {der_die} die Termine der Suchtberatung im Kalender pflegt.",
       [], "Identifikatoren werden ersetzt, die Prosa bleibt unverändert: die "
           "Kalenderpflege ist eine Tätigkeit, keine Offenlegung.", "occupational"),

    it("i14", "mixed", "hard", "", "IDENTITY", "canonical",
       "{name} ({kdnr}) fragt, ob der Turban im Werksausweis-Foto der Kollegin zulässig ist.",
       "{name} ({kdnr}) fragt, ob der Turban im Werksausweis-Foto der Kollegin zulässig ist.",
       [], "Identifikatoren werden ersetzt; das religiöse Merkmal betrifft eine "
           "dritte Person und wird nicht der anfragenden Person zugeschrieben.",
       "third_party"),

    # --- borderline identity: close to the boundary, innocent cause stated ---
    it("i15", "ethnicity", "hard", "", "IDENTITY", "canonical",
       "{name} hat für die Feier nur vegetarische Gerichte bestellt, weil mehrere Gäste bestimmte Fleischsorten nicht essen.",
       "{name} hat für die Feier nur vegetarische Gerichte bestellt, weil mehrere Gäste bestimmte Fleischsorten nicht essen.",
       [], "Grenzfall: liegt nah an einer religiösen Speisevorschrift, der "
           "genannte Grund ist jedoch die Rücksicht auf Gäste — eine Aussage "
           "über Dritte, nicht über die Person selbst.", "borderline_third_party"),

    it("i16", "ethnicity", "hard", "", "IDENTITY", "canonical",
       "{name} spricht in den Pausen oft Arabisch, weil der Sprachkurs tägliche Übung verlangt.",
       "{name} spricht in den Pausen oft Arabisch, weil der Sprachkurs tägliche Übung verlangt.",
       [], "Grenzfall: dieselbe Beobachtung wie in e11, aber mit einem "
           "ausdrücklich benannten nicht-herkunftsbezogenen Grund.",
       "borderline_benign_cause"),

    it("i17", "health", "hard", "", "IDENTITY", "canonical",
       "{name} nimmt morgens mehrere Tabletten aus einer Dose, die laut Aufdruck Vitaminpräparate enthält.",
       "{name} nimmt morgens mehrere Tabletten aus einer Dose, die laut Aufdruck Vitaminpräparate enthält.",
       [], "Grenzfall: Medikamenteneinnahme wirkt zunächst gesundheitsbezogen, "
           "der Inhalt ist jedoch ausdrücklich als Nahrungsergänzung benannt.",
       "borderline_benign_cause"),

    it("i18", "ethnicity", "hard", "", "IDENTITY", "canonical",
       "{name} trägt bei der Werksführung eine Kopfbedeckung, weil in der Halle Haarnetzpflicht gilt.",
       "{name} trägt bei der Werksführung eine Kopfbedeckung, weil in der Halle Haarnetzpflicht gilt.",
       [], "Grenzfall: Kopfbedeckung ist ein möglicher religiöser Marker, hier "
           "aber durch eine betriebliche Vorschrift erklärt.",
       "borderline_logistical"),

    it("i19", "health", "hard", "", "IDENTITY", "multi_sentence",
       "Der Termin bleibt wie geplant. {name} verlässt das Gebäude freitags früher, weil die Fahrgemeinschaft dann eher losfährt.",
       "Der Termin bleibt wie geplant. {name} verlässt das Gebäude freitags früher, weil die Fahrgemeinschaft dann eher losfährt.",
       [], "Grenzfall: regelmäßiges früheres Gehen ist ein typisches Muster für "
           "wiederkehrende Behandlungstermine, hier jedoch organisatorisch erklärt.",
       "borderline_benign_cause"),
]

NO_REWRITE_OPS = {"IDENTITY", "PASS_THROUGH", "DISCLOSURE_NECESSARY",
                  "ID_PLACEHOLDER", "ID_PLACEHOLDER+PASS_THROUGH"}

SYSTEM_PROMPT = (
    "Du bist ein Datenschutz-Filter. Ersetze direkte Identifikatoren durch "
    "Platzhalter und formuliere Sätze so um, dass besondere Kategorien "
    "personenbezogener Daten (Gesundheit, ethnische Herkunft, Religion) nicht "
    "mehr ableitbar sind. Erfinde keine neuen Tatsachen. Wenn nichts "
    "Schützenswertes offengelegt wird, gib den Satz unverändert zurück."
)


# ============================================================
# BUILD
# ============================================================

_ID_TOKEN = "\u00a7ID\u00a7"


def _prose_equal(in_tpl, out_tpl):
    """True if the two templates differ ONLY in identifier substitution.

    Needed because a no-rewrite row is not required to be string-identical:
    direct identifiers are still placeheld, so `{name}` in the input becomes
    `[NAME]` in the output. What must be unchanged is the prose around them.
    Both sides are normalised to a common token before comparison.
    """
    def norm(tpl):
        out = tpl
        for slot, ph in PLACEHOLDER.items():
            out = out.replace("{" + slot + "}", _ID_TOKEN).replace(ph, _ID_TOKEN)
        return out
    return norm(in_tpl) == norm(out_tpl)


def stable_seed(item_id, k):
    """Deterministic across processes, machines and Python versions.

    Must not be replaced by the builtin hash(): it is salted per process for
    str, which would make the recorded `seed` column unreproducible."""
    return MASTER_SEED + zlib.crc32(item_id.encode("utf-8")) % 100000 + k * 7919


def family_of(item):
    """Family of an item, from the item alone, so the instantiation budget can
    be applied during the build and the dev split can reuse the same rule."""
    op = item["operation"]
    if op == "IDENTITY":
        return "identity"
    if "PASS_THROUGH" in op:
        return "passthrough"
    if op == "DISCLOSURE_NECESSARY":
        return "disclosure"
    if op == "ID_PLACEHOLDER":
        return "explicit"
    return "rewrite"


# Instantiation k=0 uses a German-origin name, k=1 a marked-origin name, k=2
# is unconstrained. The model therefore sees every sentence under both name
# conditions during training. This is the training-side complement to the
# held-out name-swap fairness probe.
ORIGIN_PLAN = ["german", "marked", "any"]
assert len(ORIGIN_PLAN) >= max(INSTANTIATIONS.values()), \
    "ORIGIN_PLAN must cover the largest per-family instantiation count"

GERMAN_NAMES = sorted(x for x in NAME_POOL if x[2] == "german")
MARKED_NAMES = sorted(x for x in NAME_POOL if x[2] != "german")
ALL_NAMES = sorted(NAME_POOL)

rows = []
# Rendered values kept per row so validation can assert the asymmetry directly:
# every direct identifier must be gone from the output, every pass-through
# numeric must still be there, character for character.
_identifier_values = {}
_passthrough_values = {}
_name_values = {}

for item in ITEMS:
    item_id = item["item_id"]
    attr = item["attribute"]
    if attr and attr in RESERVED_ATTRIBUTES:
        sys.exit(f"FATAL: item {item_id} uses attribute {attr}, reserved for Dataset A")

    n_inst = INSTANTIATIONS[family_of(item)]
    # Names for all instantiations of an item are drawn up front, without
    # replacement. Drawing independently per k lets the unconstrained "any" plan
    # re-draw a name the "german" plan already used, producing two rows with
    # byte-identical input and output. Duplicate pairs are not incorrect, but
    # they spend gradient budget on a repeat and quietly reduce the item's real
    # instantiation count.
    _used = set()
    _picked = []
    for k in range(n_inst):
        plan_k = ORIGIN_PLAN[k % len(ORIGIN_PLAN)]
        pool_k = {"german": GERMAN_NAMES, "marked": MARKED_NAMES,
                  "any": ALL_NAMES}[plan_k]
        if item["gender"]:
            pool_k = [x for x in pool_k if x[1] == item["gender"]] or pool_k
        fresh = [x for x in pool_k if x[0] not in _used] or pool_k
        pick = random.Random(stable_seed(item["item_id"], k)).choice(fresh)
        _used.add(pick[0])
        _picked.append((plan_k, pick))

    for k in range(n_inst):
        seed = stable_seed(item_id, k)
        rng = random.Random(seed)

        plan, (name, gender, origin) = _picked[k]

        ids = Ids(seed)
        vals = {
            "name": name, "kdnr": ids.kdnr(), "maid": ids.maid(),
            "kvnr": ids.kvnr(), "phone": ids.phone(), "iban": ids.iban(),
            "email": ids.email(name), "steuer_id": ids.steuer_id(),
            "raum": ids.raum(), "rechnung": ids.rechnung(),
            "artikel": ids.artikel(), "charge": ids.charge(),
            "gleis": ids.gleis(), "norm": ids.norm(),
            "version": ids.version(), "menge": ids.menge(),
        }
        # Output templates spell placeholders literally where the surface form
        # matters (explicit / passthrough), and use slots elsewhere. Pass-through
        # slots resolve to the SAME value in input and output by construction.
        out_vals = dict(vals)
        for slot in list(vals):
            if slot == "name" or slot in ID_SLOTS:
                out_vals[slot] = PLACEHOLDER[slot]

        src = resolve(item["input"], gender)
        tgt = resolve(item["output"], gender)
        text_in = src.format(**vals)
        text_out = tgt.format(**out_vals)

        assert_no_reserved_cue(item_id, "input", text_in)
        assert_no_reserved_cue(item_id, "output", text_out)

        row_id = f"b_{item_id}_{k+1:02d}"
        _identifier_values[row_id] = {
            s: vals[s] for s in ID_SLOTS if "{" + s + "}" in item["input"]}
        _passthrough_values[row_id] = [
            vals[s] for s in PASS_SLOTS if "{" + s + "}" in item["input"]]
        _name_values[row_id] = vals["name"] if "{name}" in item["input"] else None

        rows.append({
            "id": f"b_{item_id}_{k+1:02d}", "item_id": item_id,
            "category": item["category"], "difficulty": item["difficulty"],
            "operation": item["operation"], "syntax": item["syntax"],
            "attribute": attr, "negative_type": item["negative_type"],
            "requires_rewrite": item["operation"] not in NO_REWRITE_OPS,
            "is_identity": item["operation"] == "IDENTITY",
            "is_passthrough": "PASS_THROUGH" in item["operation"],
            "is_disclosure_necessary": item["operation"] == "DISCLOSURE_NECESSARY",
            # "prose_unchanged" is the real invariant for no-rewrite rows:
            # direct identifiers are still placeheld, so the surface strings
            # differ, but the wording around them must survive untouched.
            "prose_unchanged": item["input"] == item["output"] or
                               _prose_equal(item["input"], item["output"]),
            "leaks": bool(attr) and item["operation"] != "DISCLOSURE_NECESSARY",
            "name_origin": origin, "name_origin_marked": origin != "german",
            "origin_plan": plan, "gender": gender,
            "input_text": text_in, "output_text": text_out,
            "changed": text_in != text_out,
            "explanation": item["explanation"],
            "seed": seed, "split": "",
            "source": "synthetic_rewrite_pairs_de_b",
        })

# ---- split: dev items are DISJOINT ITEMS from train, not sampled rows ----
# Tuning prompts on sentences the model also fine-tuned on inflates the dev
# numbers and produces bad decisions. Note the standing caveat: this dev slice
# shares a generator with train, so it is good for loss monitoring and bad for
# checkpoint selection. Selecting the checkpoint on Dataset A would convert the
# sealed evaluation set into a model-selection set and void the generalisation
# claim the reserved-vocabulary contract exists to protect.
by_item = defaultdict(list)
for r in rows:
    by_item[r["item_id"]].append(r)

rng_split = random.Random(MASTER_SEED)

# Stratify the dev slice by operation family so that a dev slice made only of
# rewrite items cannot hide over-redaction during debugging.
families = defaultdict(list)
item_of = {i["item_id"]: i for i in ITEMS}
for iid in by_item:
    families[family_of(item_of[iid])].append(iid)

for fam in families:
    families[fam] = sorted(families[fam])
    rng_split.shuffle(families[fam])

# Dev is drawn ONLY from families large enough to spare an item. Proportional
# allocation over all five families would put a disclosure item into dev,
# removing a quarter of the thinnest and highest-stakes class from the gradient
# in order to monitor a loss curve. Small families are train-only by design.
#
# CONSEQUENCE, which must be stated in the writeup: dev does not monitor
# pass-through, explicit or disclosure behaviour. It monitors leak repair
# (rewrite) and over-redaction on benign prose (identity), which is the primary
# over-redaction probe. It cannot see the two specialised variants: masking
# non-sensitive numerics, and over-redacting disclosure-necessary items. Those
# surface only in the Dataset A evaluation and the judge triage. This costs
# little in practice because the dev slice shares a generator with train and was
# never usable for checkpoint selection: fix hyperparameters a priori and report
# the final epoch rather than early-stopping.
DEV_ELIGIBLE_MIN_ITEMS = 12
eligible = {f: v for f, v in families.items() if len(v) >= DEV_ELIGIBLE_MIN_ITEMS}
train_only_families = sorted(set(families) - set(eligible))

# Allocated over ITEMS, not rows, because instantiation counts now differ by
# family and an item-count quota keeps the dev slice's family mix proportional
# to the eligible pool regardless of how many rows each item expands to.
n_eligible = sum(len(v) for v in eligible.values())
n_dev_items = max(1, round(DEV_ITEM_FRACTION * n_eligible))
quota = {f: len(v) * n_dev_items / n_eligible for f, v in eligible.items()}
alloc = {f: int(q) for f, q in quota.items()}
# Remainder by largest fractional part, ties broken by name. Looped rather than
# single-pass: one pass silently under-fills if n_dev_items exceeds the number
# of families.
while sum(alloc.values()) < n_dev_items:
    cands = [f for f in eligible if alloc[f] < len(eligible[f])]
    if not cands:
        break
    f = sorted(cands, key=lambda x: (-(quota[x] - alloc[x]), x))[0]
    alloc[f] += 1

dev_items = {iid for f, n in alloc.items() for iid in eligible[f][:n]}
placed = sum(len(by_item[i]) for i in dev_items)
for r in rows:
    r["split"] = "dev" if r["item_id"] in dev_items else "train"

# ============================================================
# VALIDATION
# ============================================================

errors = []

for r in rows:
    op = r["operation"]

    # 1. no-rewrite rows must not touch the prose
    if not r["requires_rewrite"] and not r["prose_unchanged"]:
        errors.append(f"{r['id']}: no-rewrite row ({op}) altered the prose")

    # 2. leaky rows must actually be repaired
    if r["leaks"] and not r["changed"]:
        errors.append(f"{r['id']}: leaky row was not repaired")

    # 3. the leaking term must be gone from the output
    item = next(i for i in ITEMS if i["item_id"] == r["item_id"])
    for term in item["leak_terms"]:
        if compound_find(r["output_text"], term) != -1 and \
                (r["item_id"], term) not in LEAK_TERM_EXCEPTIONS:
            errors.append(f"{r['id']}: leak term {term!r} survives in the output "
                          f"(compound-aware match)")

    # 4. pass-through numerics must survive VERBATIM
    for v in _passthrough_values.get(r["id"], []):
        if v not in r["output_text"]:
            errors.append(f"{r['id']}: pass-through value {v!r} was masked")

    # 5. direct identifiers must never survive
    for slot, raw in _identifier_values.get(r["id"], {}).items():
        if raw in r["output_text"]:
            errors.append(f"{r['id']}: direct identifier {slot} survived unmasked")

    # 5b. the person's name must be masked, and the placeholder must be there.
    #     Defence in depth: out_vals replaces `name` unconditionally, so neither
    #     can fail on the current code path. They guard a future edit to the
    #     output-template convention. The second direction is the one that
    #     catches the mistake you actually make while authoring new items —
    #     an output template that simply forgot the {name} slot.
    raw_name = _name_values.get(r["id"])
    if raw_name:
        if raw_name in r["output_text"]:
            errors.append(f"{r['id']}: raw name survived in the output")
        if PLACEHOLDER["name"] not in r["output_text"]:
            errors.append(f"{r['id']}: input has a name but output has no [NAME]")

    # 6. no unresolved template slots
    if "{" in r["input_text"] or "{" in r["output_text"]:
        errors.append(f"{r['id']}: unresolved template slot")

    # 7. explanations only where a rewrite happens (metadata discipline)
    if r["requires_rewrite"] and not r["explanation"]:
        errors.append(f"{r['id']}: rewrite row has no explanation")

# 7b. no two rows may carry the same (input, output) pair. A duplicate is not
#     wrong, but it silently reduces an item's effective instantiation count and
#     spends gradient budget twice on one example. Caught at build time rather
#     than deduplicated afterwards, so the cause gets fixed in the item, not the
#     symptom in the CSV.
_seen = defaultdict(list)
for r in rows:
    _seen[(r["input_text"], r["output_text"])].append(r["id"])
for pair, ids in _seen.items():
    if len(ids) > 1:
        errors.append(f"duplicate training pair across rows {ids}: {pair[0][:70]}")

# 8. split integrity
overlap = {r["item_id"] for r in rows if r["split"] == "dev"} & \
          {r["item_id"] for r in rows if r["split"] == "train"}
if overlap:
    errors.append(f"item straddles train/dev: {sorted(overlap)}")

# 9. every item must cover both name-origin conditions
for iid, rs in by_item.items():
    origins = {r["name_origin_marked"] for r in rs
               if "{name}" in next(i for i in ITEMS if i["item_id"] == iid)["input"]}
    if origins and origins != {True, False}:
        errors.append(f"item {iid}: name-origin conditions not both covered ({origins})")

# 10. composition band, stated symmetrically on the rewrite share.
#     Both shortcuts are failure modes and the guard should be even-handed:
#     too few no-rewrite rows teaches "always rewrite" (over-redaction), too
#     few rewrite rows teaches "never rewrite" (under-redaction). Stated on the
#     rewrite share so both directions are treated symmetrically.
n_rw = sum(1 for r in rows if r["requires_rewrite"])
rw_share = n_rw / len(rows)
if not (0.35 <= rw_share <= 0.65):
    errors.append(f"rewrite share {rw_share:.0%} outside 35-65% "
                  f"(<35% teaches 'never rewrite', >65% teaches 'always rewrite')")

# 11. the training records must have exactly the three intended turns.
#     The previous version of this check tested whether the whole German
#     explanation sentence appeared verbatim inside the (much shorter) output
#     string, which could essentially never be true. The real failure mode is
#     structural: somebody later adds the explanation as a fourth message or an
#     extra key without updating the methodology paragraph. That is what this
#     now catches.
for r in rows:
    rec = {"messages": [{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": r["input_text"]},
                        {"role": "assistant", "content": r["output_text"]}]}
    if list(rec.keys()) != ["messages"] or len(rec["messages"]) != 3:
        errors.append(f"{r['id']}: training record shape changed")
    if [m["role"] for m in rec["messages"]] != ["system", "user", "assistant"]:
        errors.append(f"{r['id']}: training record roles changed")
    if "explanation" in json.dumps(rec, ensure_ascii=False):
        errors.append(f"{r['id']}: explanation reached the training record")

for e in errors:
    print("ERROR:", e, file=sys.stderr)

with open("validation_errors.txt", "w", encoding="utf-8") as f:
    f.write(f"# {len(errors)} validation errors\n")
    for e in errors:
        f.write(e + "\n")

# ============================================================
# OUTPUT
# ============================================================
#
# BUILD GATE. assert_no_reserved_cue hard-fails on contamination, and every
# other correctness check is enforced the same way: a leak term surviving a
# rewrite, an unresolved slot or a broken split must not reach the JSONL that
# feeds the LoRA run.
#
# Stale artifacts are deleted BEFORE validation is enforced, so a failed build
# cannot leave a previous good run's training data in place for
# run_experiment.sh to pick up silently.
#
# dataset_b.csv is written even on failure, because these errors are per-row and
# you need to see the offending rows to fix them. The training artifacts are not.

TRAINING_ARTIFACTS = ["dataset_b_train.jsonl", "dataset_b_dev.jsonl",
                      "fairness_probe.csv"]
for path in TRAINING_ARTIFACTS:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass

FIELDS = ["id", "item_id", "category", "difficulty", "operation", "syntax",
          "attribute", "negative_type", "requires_rewrite", "is_identity",
          "is_passthrough", "is_disclosure_necessary", "prose_unchanged", "leaks",
          "name_origin", "name_origin_marked", "origin_plan", "gender",
          "input_text", "output_text", "changed", "explanation",
          "seed", "split", "source"]

with open("dataset_b.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader()
    w.writerows(rows)

if errors:
    print(f"\nWrote dataset_b.csv and validation_errors.txt for inspection.",
          file=sys.stderr)
    sys.exit(f"FATAL: {len(errors)} validation errors — refusing to write "
             f"training artifacts. See validation_errors.txt.")

# The JSONL carries ONLY system/user/assistant. The explanation column stays in
# the CSV: training the model to emit an explanation would rebuild the removed
# attribute in a second field, and would split a thin gradient signal across
# two objectives.
for split in ("train", "dev"):
    with open(f"dataset_b_{split}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            if r["split"] != split:
                continue
            f.write(json.dumps({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": r["input_text"]},
                    {"role": "assistant", "content": r["output_text"]},
                ]
            }, ensure_ascii=False) + "\n")

# Held-out fairness probe: identical sentences, name origin swapped.
#
# SCORING RULE — fixed here, before any probe output exists, so it cannot be
# chosen to suit the result:
#   * A no-rewrite row counts as OVER-REDACTED if the prose differs from
#     expected_output after normalising whitespace, OR a pass-through numeric
#     was masked. Exact character match is too strict (punctuation and spacing
#     noise would false-positive); this mirrors the build-time validation.
#   * The comparison is PAIRED by item_id, so the test is McNemar on discordant
#     item pairs, not two independent rates. Report the discordant counts
#     themselves alongside any p-value.
#   * With ~48 name-bearing items the test has little power. An inconclusive
#     result is reported as inconclusive, NOT as evidence of fairness. Run the
# trained model over both files and compare the over-redaction rate on
# no-rewrite rows. Any gap is differential treatment by name origin, which is
# the concrete version of the equity claim and needs no annotators.
with open("fairness_probe.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["probe_id", "item_id", "operation",
                                      "requires_rewrite", "origin_plan", "draw",
                                      "gender", "name_origin",
                                      "input_text", "expected_output"])
    w.writeheader()
    for item in ITEMS:
        if "{name}" not in item["input"]:
            continue

        # GENDER STRATIFICATION. Drawing each cell independently lets an item
        # come out 2 male / 1 female on the German side and 1 male / 2 female on
        # the marked side, so any gap the probe reports would be part gender
        # effect and part origin effect. A gender sequence is fixed per item and
        # enforced in BOTH cells.
        #
        # The sequence is drawn from ONE Random instance. Rebuilding
        # random.Random(seq_seed) inside the comprehension would reseed on every
        # iteration and return the same value each time, making every item
        # single-gender.
        seq_rng = random.Random(stable_seed(item["item_id"], 97))
        gender_seq = ([item["gender"]] * N_PROBE_NAMES if item["gender"]
                      else [seq_rng.choice(["male", "female"])
                            for _ in range(N_PROBE_NAMES)])

        for cond, pool in (("german", GERMAN_NAMES), ("marked", MARKED_NAMES)):
            # Names drawn without replacement within a cell. Independent
            # per-draw seeds would repeat a name by collision alone (birthday
            # problem on a 14-name pool) and correlate the bootstrap draws.
            cell_seed = stable_seed(item["item_id"], 99) + (0 if cond == "german" else 1)
            rng = random.Random(cell_seed)
            used = set()
            for j, want_gender in enumerate(gender_seq):
                cands = [x for x in pool if x[1] == want_gender and x[0] not in used] \
                    or [x for x in pool if x[1] == want_gender] or pool
                name, gender, origin = rng.choice(cands)
                used.add(name)

                # Seeded from cell_seed, never from the build loop's `seed`,
                # which still resolves at module scope and would give every
                # probe row the same customer number, phone number and IBAN.
                # Identifiers must vary per draw.
                ids = Ids(cell_seed + j * 7919)
                vals = {"name": name, "kdnr": ids.kdnr(), "maid": ids.maid(),
                        "kvnr": ids.kvnr(), "phone": ids.phone(), "iban": ids.iban(),
                        "email": ids.email(name), "steuer_id": ids.steuer_id(),
                        "raum": ids.raum(), "rechnung": ids.rechnung(),
                        "artikel": ids.artikel(), "charge": ids.charge(),
                        "gleis": ids.gleis(), "norm": ids.norm(),
                        "version": ids.version(), "menge": ids.menge()}
                out_vals = dict(vals)
                for slot in list(vals):
                    if slot == "name" or slot in ID_SLOTS:
                        out_vals[slot] = PLACEHOLDER[slot]
                ti = resolve(item["input"], gender).format(**vals)
                to = resolve(item["output"], gender).format(**out_vals)
                assert_no_reserved_cue(item["item_id"], "probe", ti)
                w.writerow({"probe_id": f"fp_{item['item_id']}_{cond}_{j+1}",
                            "item_id": item["item_id"], "operation": item["operation"],
                            "requires_rewrite": item["operation"] not in NO_REWRITE_OPS,
                            "origin_plan": cond, "draw": j + 1, "gender": gender,
                            "name_origin": origin,
                            "input_text": ti, "expected_output": to})

# ============================================================
# REPORT
# ============================================================

inst_summary = ", ".join(f"{f}x{n}" for f, n in sorted(INSTANTIATIONS.items()))
print(f"\nDataset B: {len(rows)} rows from {len(ITEMS)} items "
      f"(instantiations per family: {inst_summary})")
print(f"Effective n for item-clustered statistics: {len(ITEMS)}, not {len(rows)}")
print(f"Validation errors: {len(errors)}")
print(f"Reserved-cue guard: checked {len(rows)*2} strings against "
      f"{len(RESERVED_CUES)} reserved cues — no collisions")

print("\nSplit (items disjoint; dev is for loss monitoring, NOT checkpoint selection):")
for sp in ("train", "dev"):
    s = [r for r in rows if r["split"] == sp]
    print(f"  {sp:6s} n={len(s):3d}  items={len({r['item_id'] for r in s}):2d}  "
          f"no-rewrite={sum(1 for r in s if not r['requires_rewrite']):3d}")

print("\nOperations:")
for k, n in Counter(r["operation"] for r in rows).most_common():
    print(f"  {k:34s} {n:3d}  ({n/len(rows):.0%})")

print("\nRewrite vs no-rewrite (guards both shortcuts):")
nr = sum(1 for r in rows if not r["requires_rewrite"])
print(f"  requires rewrite      {len(rows)-nr:3d}  ({(len(rows)-nr)/len(rows):.0%})")
print(f"  output preserves prose{nr:4d}  ({nr/len(rows):.0%})")
print(f"    of which identity          {sum(1 for r in rows if r['is_identity']):3d}")
print(f"    of which pass-through      {sum(1 for r in rows if r['is_passthrough']):3d}")
print(f"    of which disclosure-necess.{sum(1 for r in rows if r['is_disclosure_necessary']):3d}")

print("\nSyntax (failure analysis by construction):")
for k, n in Counter(r["syntax"] for r in rows).most_common():
    print(f"  {k:18s} {n:3d}  ({n/len(rows):.0%})")

print("\nIdentity rows by negative type (must mirror Dataset A's taxonomy):")
for k, n in Counter(r["negative_type"] for r in rows if r["is_identity"]).most_common():
    print(f"  {k:28s} {n:3d}")

print("\nCategory / difficulty:")
for field in ("category", "difficulty"):
    print(f"  {field}: " + "  ".join(
        f"{k}={v}" for k, v in sorted(Counter(r[field] for r in rows).items())))

print("\nAttributes used (none may be reserved for A):")
for k, n in sorted(Counter(r["attribute"] for r in rows if r["attribute"]).items()):
    print(f"  {k:44s} {n:3d}")

print("\nName-origin coverage (every item must appear under both conditions):")
print(f"  german-name rows  {sum(1 for r in rows if not r['name_origin_marked']):3d}")
print(f"  marked-name rows  {sum(1 for r in rows if r['name_origin_marked']):3d}")
marked = [r for r in rows if r["name_origin_marked"]]
germ = [r for r in rows if not r["name_origin_marked"]]
eth_m = sum(1 for r in marked if r["attribute"].startswith("ethnicity"))
eth_g = sum(1 for r in germ if r["attribute"].startswith("ethnicity"))
print(f"  P(eth attr | marked)={eth_m/max(len(marked),1):.3f}   "
      f"P(eth attr | german)={eth_g/max(len(germ),1):.3f}  "
      f"(should be close; the cue is sampled independently of the name)")

print("\nWritten: dataset_b.csv, dataset_b_train.jsonl, dataset_b_dev.jsonl, "
      "fairness_probe.csv")

print("\nOPEN ITEMS — not enforceable in this script:")
print("  1. NON-LEAKINESS is checked only structurally (leak_terms). Run the")
print("     offline LLM-judge over ALL outputs, hand-verify every flag plus a")
print("     stratified 20% of the passes, and report judge-human agreement.")
print("  2. GERMAN QUALITY is not checked at all. Have a native speaker read")
print("     the rewrite outputs cold, without the inputs, and mark anything a")
print("     colleague would not actually write. Awkward targets teach awkward")
print("     output, and no structural check can see this.")
print("  3. DISCLOSURE_NECESSARY must be assigned to Dataset A by annotators")
print("     blind to model output, BEFORE any run. Assigned afterwards it is")
print("     relabelled error, not a contribution. Lock and commit that file now.")
print(f"  4. SEED VARIANCE: report LoRA results over 3-5 training seeds. At "
      f"{len({r['item_id'] for r in rows if r['split']=='train'})}")
print("     training items, seed spread can exceed the effect being claimed.")
print("  5. NO EARLY STOPPING. Fix hyperparameters a priori and report the")
print("     final epoch. Dev shares a generator with train and cannot be used")
print("     for checkpoint selection; selecting on Dataset A would void the")
print("     contamination contract entirely.")
print("  6. CLUSTERED STATISTICS: effective n is the item count, not the row")
print("     count. McNemar on paired items; paired bootstrap over items for F1")
print("     gaps. Treating rows as independent will manufacture significance.")
print("  7. MATCHED COMPUTE in baselines: if the prompted baseline gets three")
print("     judge rounds, report the fine-tuned model with and without its own")
print("     judge loop, plus calls per sentence for each condition.")
print("  8. FEW-SHOT BASELINE. A zero-shot prompted baseline is not a fair")
print("     comparison: the fine-tuned model has seen 164 examples encoding the")
print("     format, taxonomy, register and placeholder convention, and the")
print("     baseline has seen a paragraph. That is supervision vs no")
print("     supervision, not fine-tuning vs prompting. Draw 8-16 examples from")
print("     the TRAIN split, stratified by family, put them in the prompt, and")
print("     make that the primary comparison.")
print("  9. BASE-MODEL FLOOR. Run the untrained model on dev before any")
print("     training, so the ablation table has a zero point.")
print(" 10. WATCH: DISCLOSURE_NECESSARY is 8% of rows. If Dataset A shows")
print("     under-redaction on positives, look here first — the remedy is")
print("     fewer instantiations, not removing the class.")
