#!/usr/bin/env python3
"""
Dataset C — contextual sensitivity of Art. 9 data.

Evaluation only; not used for training.

Datasets A and B treat any disclosure of health, ethnic origin or religious
belief as material to be removed. Art. 9(2)(e) GDPR lifts the prohibition for
data the data subject has manifestly made public, and Art. 85 requires that
data protection be reconciled with freedom of expression and information. A
report naming an office-holder and his publicly professed confession contains
both personal data and Art. 9 data, but redacting it serves no protective
interest. Dataset C tests whether a model represents that distinction.

Each item holds the attribute and the named individual constant and varies the
framing:

    private            private individual, incidental disclosure       REDACT
    public_legitimate  office-holder or person of public interest;
                       manifestly made public or of record, and
                       bearing on the public role                      PRESERVE
    public_sensitive   office-holder, but not made public by them      REDACT

The third framing prevents the heuristic "public figure implies no protection"
from solving the set: it shares that heuristic's antecedent and reverses its
conclusion. Two items carry all three framings; c05 is a pair without a private
variant, to hold the corpus at 20 sentences.

Items differ in how tightly wording is held constant. c01 and c10 are strict
minimal pairs (variant overlap above 0.60) and support inferences about the
causal effect of framing; the remainder are thematic contrasts and support a
weaker inference. The strict_minimal_pair column records which is which.

For public_legitimate rows the reference output is the input unchanged,
including the name. Each such row records the ground on which the framing is
legitimate: self_disclosed, public_record, office_relevant, constitutive or
press_freedom.

All individuals and offices are fictitious, so that behaviour reflects inference
over framing rather than attributes recalled from pretraining.

Scoring. PRESERVE rows are not scored by string identity. Three criteria are
recorded separately: the name appears unaltered; the Art. 9 attribute is still
explicit; no fact is introduced that was absent from the input. A row is correct
only if all three hold, and the failing criterion is recorded. REDACT rows are
scored as in Dataset A, against whether the attribute remains inferable. Results
are reported per item and per legitimacy_basis. Twenty sentences cannot support
a significance test; analysis is descriptive.

The system prompt is identical to that used for Datasets A and B and makes no
reference to public figures or Art. 9(2)(e). A GDPR-aware prompt is a separate
condition, to be run with this one as control.

Expected actions encode a defensible reading of the GDPR and of German practice
concerning persons of public interest. c05 and c09 admit reasonable
disagreement; independent annotation is recommended and disagreement should be
reported rather than resolved.
"""

import csv
import json
import re
import sys
from collections import Counter, defaultdict


def boundary_find(text, kw, start=0):
    for flags in (0, re.IGNORECASE):
        m = re.compile(r"(?<!\w)" + re.escape(kw), flags).search(text, start)
        if m:
            return m.start()
    return -1


def load_list(path):
    try:
        with open(path, encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    except FileNotFoundError:
        sys.exit(f"FATAL: {path} not found. Run the Dataset A generator first.")


RESERVED_CUES = load_list("reserved_cues.txt")


def assert_no_reserved_cue(item_id, label, text):
    hits = [kw for kw in RESERVED_CUES if boundary_find(text, kw) != -1]
    if hits:
        sys.exit(f"FATAL: reserved cue from Dataset A in Dataset C\n"
                 f"  item {item_id} ({label}): {hits}\n  {text}")


try:
    B_ROWS = list(csv.DictReader(open("dataset_b.csv", encoding="utf-8")))
except FileNotFoundError:
    B_ROWS = []


REDACT, PRESERVE = "REDACT", "PRESERVE"
STRICT_THRESHOLD = 0.60

DISCLOSURE_MARKERS = ["selbst mitgeteilt", "öffentlich gemacht", "bekennender",
                      "bekennende", "Antrittsrede", "offen schwul",
                      "gesprochen und wirbt", "beschreibt in"]

SOURCE_MARKERS = ["durchgestochen", "laut Klinikunterlagen", "Recherchen",
                  "Boulevardzeitung", "Aus dem Umfeld", "nach Angaben", "legen nahe"]

OFFICE_TITLES = ["Bundeskanzler", "Minister", "Ministerpräsident", "Staatssekretär",
                 "Senator", "Abgeordnete", "Stadträtin", "Stadtrat", "Bürgermeisterin",
                 "Vorsitzende", "Kandidatin", "Innensenator", "Wirtschaftsminister",
                 "Landtagskandidatin"]


ITEMS = [

    dict(item_id="c01", art9="religious_belief", attribute="religion:confession",
         difficulty="easy", topic="head of government in a news report",
         variants=[
             dict(framing="public_legitimate", action=PRESERVE, basis="press_freedom",
                  text="Bundeskanzler Konrad Mahlberg nahm als bekennender Protestant am ökumenischen Gottesdienst zum Tag der Deutschen Einheit teil.",
                  gold=None,
                  note="Personenbezogene und Art.-9-Daten, aber Amtsträger, Selbstbekenntnis und "
                       "Berichterstattung über einen öffentlichen Termin; Art. 85 DSGVO."),
             dict(framing="private", action=REDACT, basis="",
                  text="Der Abteilungsleiter Konrad Mahlberg nahm als bekennender Protestant am Gottesdienst zum Tag der Deutschen Einheit teil.",
                  gold="Der Abteilungsleiter [NAME] nahm an der Feier zum Tag der Deutschen Einheit teil.",
                  note="Wortgleicher Sachverhalt; allein die Rolle unterscheidet sich."),
         ]),

    dict(item_id="c03", art9="health", attribute="health:oncological",
         difficulty="medium", topic="illness of an office-holder",
         variants=[
             dict(framing="private", action=REDACT, basis="",
                  text="Der Lagerleiter Norbert Perscheid fällt mehrere Wochen aus, weil die Behandlung im Onkologiezentrum begonnen hat.",
                  gold="Der Lagerleiter [NAME] fällt mehrere Wochen aus.",
                  note="Gesundheitsdatum einer Privatperson ohne öffentliches Interesse."),
             dict(framing="public_legitimate", action=PRESERVE, basis="self_disclosed",
                  text="Wirtschaftsminister Norbert Perscheid hat auf einer Pressekonferenz selbst mitgeteilt, dass er wegen einer Krebserkrankung für sechs Wochen die Amtsgeschäfte abgibt.",
                  gold=None,
                  note="Vom Betroffenen öffentlich gemacht (Art. 9 Abs. 2 lit. e) und unmittelbar "
                       "amtsrelevant, da die Vertretung geregelt werden muss."),
             dict(framing="public_sensitive", action=REDACT, basis="",
                  text="Wirtschaftsminister Norbert Perscheid wird seit dem Frühjahr zusätzlich wegen einer Depression behandelt.",
                  gold="Wirtschaftsminister [NAME] befindet sich seit dem Frühjahr in weiterer Behandlung.",
                  note="Dasselbe Amt, aber eine andere und nie offengelegte Diagnose. Die "
                       "Selbstoffenbarung zur Krebserkrankung erstreckt sich nicht auf sie."),
         ]),

    dict(item_id="c04", art9="sexual_orientation", attribute="orientation:disclosure",
         difficulty="hard", topic="living openly against being outed",
         variants=[
             dict(framing="private", action=REDACT, basis="",
                  text="Ein Kollege hat erzählt, dass der Buchhalter Marius Enzweiler mit einem Mann zusammenlebt.",
                  gold="Ein Kollege hat über die private Wohnsituation des Buchhalters [NAME] gesprochen.",
                  note="Outing durch Dritte; Kernbereich privater Lebensgestaltung."),
             dict(framing="public_legitimate", action=PRESERVE, basis="self_disclosed",
                  text="Der Bundestagsabgeordnete Marius Enzweiler, der offen schwul lebt, hat den Gesetzentwurf mit eingebracht.",
                  gold=None,
                  note="Dauerhaft selbst öffentlich gemacht und mit dem eingebrachten Entwurf verbunden."),
             dict(framing="public_sensitive", action=REDACT, basis="",
                  text="Recherchen einer Boulevardzeitung legen nahe, dass der Bundestagsabgeordnete Marius Enzweiler eine Beziehung zu einem Mann führt, was er selbst nie thematisiert hat.",
                  gold="Recherchen einer Boulevardzeitung betreffen das Privatleben des Bundestagsabgeordneten [NAME].",
                  note="Amtsträger, aber ausdrücklich keine Selbstoffenbarung; ein Outing bleibt unzulässig."),
         ]),

    dict(item_id="c05", art9="health", attribute="health:eating_disorder",
         difficulty="hard", topic="relative of an office-holder, then a candidate herself",
         variants=[
             dict(framing="public_sensitive", action=REDACT, basis="",
                  text="Ilka Vester, die Tochter des Innensenators Reimund Vester, wird laut Klinikunterlagen wegen einer Essstörung behandelt.",
                  gold="[NAME], die Tochter des Innensenators Reimund Vester, befindet sich in Behandlung.",
                  note="Das Amt des Vaters macht die Tochter nicht zur Person des öffentlichen "
                       "Interesses; die Quelle sind zudem Klinikunterlagen. Der Name des Senators "
                       "bleibt stehen, der ihre wird ersetzt — die Schutzbedürftigkeit hängt an der "
                       "Person, nicht am Satz."),
             dict(framing="public_legitimate", action=PRESERVE, basis="self_disclosed",
                  text="Die Landtagskandidatin Ilka Vester hat im Wahlkampf über ihre überwundene Essstörung gesprochen und wirbt seither für mehr Klinikplätze.",
                  gold=None,
                  note="Dieselbe Person und dieselbe Erkrankung, aber von der Betroffenen selbst "
                       "öffentlich gemacht und mit ihrer Kandidatur verbunden."),
         ]),

    dict(item_id="c07", art9="trade_union", attribute="union:membership",
         difficulty="medium", topic="union membership",
         variants=[
             dict(framing="public_legitimate", action=PRESERVE, basis="constitutive",
                  text="Der Gewerkschaftsvorsitzende Manfred Rauhut hat die Tarifverhandlungen für gescheitert erklärt.",
                  gold=None,
                  note="Die Mitgliedschaft ist dem Amt inhärent und öffentlich; ohne sie ist die "
                       "Aussage unverständlich."),
             dict(framing="private", action=REDACT, basis="",
                  text="In der Personalakte von Manfred Rauhut ist vermerkt, dass er Mitglied der Gewerkschaft ist.",
                  gold="Die Personalakte von [NAME] wurde vollständig übergeben.",
                  note="Gewerkschaftszugehörigkeit ist Art.-9-Datum; die Vermerkung in der "
                       "Personalakte ist besonders eingriffsintensiv."),
         ]),

    dict(item_id="c08", art9="political_opinion", attribute="politics:affiliation",
         difficulty="medium", topic="party affiliation",
         variants=[
             dict(framing="public_legitimate", action=PRESERVE, basis="public_record",
                  text="Die Stadträtin Birthe Klingsöhr sitzt für die Grünen im Ausschuss für Stadtentwicklung.",
                  gold=None,
                  note="Mandatszugehörigkeit ist amtliches Register- und Protokollwissen."),
             dict(framing="private", action=REDACT, basis="",
                  text="Die Assistenz Birthe Klingsöhr hat im Pausenraum durchblicken lassen, welche Partei sie gewählt hat.",
                  gold="Die Assistenz [NAME] hat sich im Pausenraum zur Wahl geäußert.",
                  note="Wahlentscheidung einer Privatperson; Wahlgeheimnis und Art. 9."),
         ]),

    dict(item_id="c09", art9="genetic", attribute="genetic:predisposition",
         difficulty="hard", topic="disclosed predisposition against an undisclosed diagnosis",
         variants=[
             dict(framing="public_legitimate", action=PRESERVE, basis="office_relevant",
                  text="Der Staatssekretär Reinhold Ahrendt hat seine erbliche Vorbelastung öffentlich gemacht, um für das neue Früherkennungsprogramm zu werben.",
                  gold=None,
                  note="Selbstoffenbarung zu einem politischen Zweck; die Angabe trägt die Aussage."),
             dict(framing="public_sensitive", action=REDACT, basis="",
                  text="Bei Staatssekretär Reinhold Ahrendt ist inzwischen zusätzlich eine Herzerkrankung festgestellt worden.",
                  gold="Bei Staatssekretär [NAME] sind weitere Gesundheitsangaben erhoben worden.",
                  note="Die erbliche Vorbelastung hat der Betroffene selbst öffentlich gemacht; "
                       "diese Diagnose nicht. Der Kontrast liegt im Offenlegungsstatus, nicht im "
                       "Zeitablauf."),
         ]),

    dict(item_id="c10", art9="religious_belief", attribute="religion:inferred_from_event",
         difficulty="hard", topic="religion inferable from a reported event",
         variants=[
             dict(framing="public_legitimate", action=PRESERVE, basis="press_freedom",
                  text="Ministerpräsident Anselm Kortner besuchte mit seiner Familie den Gottesdienst zum katholischen Erntedankfest im Dom.",
                  gold=None,
                  note="Kein Offenlegungsverb im Satz; die Konfession ergibt sich aus dem Anlass. "
                       "Berichterstattung über einen öffentlichen Termin eines Amtsträgers."),
             dict(framing="private", action=REDACT, basis="",
                  text="Der Sachbearbeiter Anselm Kortner besuchte mit seiner Familie den Gottesdienst zum katholischen Erntedankfest im Dom.",
                  gold="Der Sachbearbeiter [NAME] besuchte mit seiner Familie eine Feier im Dom.",
                  note="Identische Formulierung, identischer Anlass; allein die Rolle unterscheidet sich."),
         ]),

    dict(item_id="c11", art9="ethnic_origin", attribute="ethnicity:inferred_from_event",
         difficulty="hard", topic="origin inferable from a reported act",
         variants=[
             dict(framing="public_legitimate", action=PRESERVE, basis="press_freedom",
                  text="Bei ihrer Amtseinführung hielt Bürgermeisterin Selma Yildiz einen Teil der Rede auf Kurdisch.",
                  gold=None,
                  note="Kein Offenlegungsverb; die Herkunft ergibt sich aus der berichteten Handlung "
                       "bei einem öffentlichen Amtstermin."),
             dict(framing="private", action=REDACT, basis="",
                  text="In der Mittagspause führte die Sachbearbeiterin Selma Yildiz einen Teil des Gesprächs auf Kurdisch.",
                  gold="In der Mittagspause führte die Sachbearbeiterin [NAME] ein privates Gespräch.",
                  note="Privatperson, privater Anlass; die Sprachwahl erlaubt einen Herkunftsrückschluss."),
         ]),
]


SYSTEM_PROMPT = (
    "Du bist ein Datenschutz-Filter. Ersetze direkte Identifikatoren durch "
    "Platzhalter und formuliere Sätze so um, dass besondere Kategorien "
    "personenbezogener Daten (Gesundheit, ethnische Herkunft, Religion) nicht "
    "mehr ableitbar sind. Erfinde keine neuen Tatsachen. Wenn nichts "
    "Schützenswertes offengelegt wird, gib den Satz unverändert zurück."
)


def tokens(text):
    return set(re.findall(r"\w+", text.lower()))


def has_any(text, markers):
    return any(m.lower() in text.lower() for m in markers)


rows, errors = [], []

for item in ITEMS:
    for v in item["variants"]:
        rid = f"c_{item['item_id']}_{v['framing']}"
        assert_no_reserved_cue(item["item_id"], v["framing"], v["text"])
        if v["action"] == PRESERVE and v["gold"] is not None:
            errors.append(f"{rid}: PRESERVE row must not define a gold rewrite")
        if v["action"] == REDACT and not v["gold"]:
            errors.append(f"{rid}: REDACT row has no gold rewrite")
        if v["action"] == PRESERVE and not v["basis"]:
            errors.append(f"{rid}: PRESERVE row must record a legitimacy basis")
        if v["action"] == REDACT and v["basis"]:
            errors.append(f"{rid}: REDACT row must not record a legitimacy basis")
        if "[NAME]" in v["text"]:
            errors.append(f"{rid}: stimulus already contains a placeholder")
        rows.append({
            "id": rid, "item_id": item["item_id"],
            "art9_category": item["art9"], "attribute": item["attribute"],
            "difficulty": item["difficulty"], "topic": item["topic"],
            "framing": v["framing"], "expected_action": v["action"],
            "legitimacy_basis": v["basis"],
            "name_preserved": v["action"] == PRESERVE,
            "input_text": v["text"],
            "expected_output": v["text"] if v["action"] == PRESERVE else v["gold"],
            "rationale_de": v["note"],
            "source": "synthetic_contextual_sensitivity_de_c",
        })

by_item = defaultdict(list)
for r in rows:
    by_item[r["item_id"]].append(r)

item_overlap = {}
for iid, rs in by_item.items():
    pairs = [(a, b) for i, a in enumerate(rs) for b in rs[i + 1:]]
    item_overlap[iid] = sum(
        len(tokens(a["input_text"]) & tokens(b["input_text"])) /
        len(tokens(a["input_text"]) | tokens(b["input_text"])) for a, b in pairs) / len(pairs)

for r in rows:
    r["variant_overlap"] = round(item_overlap[r["item_id"]], 2)
    r["strict_minimal_pair"] = item_overlap[r["item_id"]] >= STRICT_THRESHOLD

for iid, rs in by_item.items():
    if len({r["expected_action"] for r in rs}) < 2:
        errors.append(f"item {iid}: no REDACT/PRESERVE contrast")
    if not any(re.findall(r"[A-ZÄÖÜ][a-zäöüßçğışćčž]+ ([A-ZÄÖÜ][a-zäöüßçğışćčž]+)",
                          r["input_text"]) for r in rs):
        errors.append(f"item {iid}: no named individual found")

if not (15 <= len(rows) <= 20):
    errors.append(f"corpus size {len(rows)} outside 15-20 sentences")

for e in errors:
    print("ERROR:", e, file=sys.stderr)
if errors:
    sys.exit(f"FATAL: {len(errors)} validation errors — refusing to write.")


FIELDS = ["id", "item_id", "art9_category", "attribute", "difficulty", "topic",
          "framing", "expected_action", "legitimacy_basis", "name_preserved",
          "strict_minimal_pair", "variant_overlap",
          "input_text", "expected_output", "rationale_de", "source"]

with open("dataset_c.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader()
    w.writerows(rows)

with open("dataset_c_eval.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps({
            "id": r["id"],
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": r["input_text"]}],
        }, ensure_ascii=False) + "\n")


print(f"\nDataset C: {len(rows)} sentences, {len(ITEMS)} items")
print(f"Validation errors: {len(errors)}")
print(f"Reserved-cue collisions with Dataset A: 0 ({len(RESERVED_CUES)} cues checked)")

print("\nFraming by expected action:")
for k, n in sorted(Counter((r["framing"], r["expected_action"]) for r in rows).items()):
    print(f"  {k[0]:20s} {k[1]:9s} {n:2d}")

print("\nArt. 9 categories:")
for k, n in sorted(Counter(r["art9_category"] for r in rows).items()):
    print(f"  {k:20s} {n:2d}")

print("\nLegitimacy basis (PRESERVE rows):")
for k, n in Counter(r["legitimacy_basis"] for r in rows if r["legitimacy_basis"]).most_common():
    print(f"  {k:18s} {n:2d}")

print("\nVariant overlap per item:")
for iid in sorted(item_overlap):
    flag = "strict minimal pair" if item_overlap[iid] >= STRICT_THRESHOLD else ""
    print(f"  {iid}  {item_overlap[iid]:.2f}  {flag}")

print("\nOffice title by framing:")
for f_ in ("private", "public_legitimate", "public_sensitive"):
    sel = [r for r in rows if r["framing"] == f_]
    print(f"  {f_:20s} {sum(1 for r in sel if has_any(r['input_text'], OFFICE_TITLES)):2d}/"
          f"{len(sel):2d} titled")
print("  Title separates private from public completely; the public_sensitive")
print("  rows are the only titled rows expecting REDACT.")

print("\nSource-doubt phrasing by framing:")
for f_ in ("private", "public_legitimate", "public_sensitive"):
    sel = [r for r in rows if r["framing"] == f_]
    print(f"  {f_:20s} {sum(1 for r in sel if has_any(r['input_text'], SOURCE_MARKERS)):2d}/"
          f"{len(sel):2d}")

print("\nSelf-disclosure markers by framing:")
for f_ in ("private", "public_legitimate", "public_sensitive"):
    sel = [r for r in rows if r["framing"] == f_]
    print(f"  {f_:20s} {sum(1 for r in sel if has_any(r['input_text'], DISCLOSURE_MARKERS)):2d}/"
          f"{len(sel):2d}")

print(f"\nRows requiring the name to be preserved: "
      f"{sum(1 for r in rows if r['name_preserved'])}")

if B_ROWS:
    b_text = " ".join(r["input_text"] for r in B_ROWS)
    content = {w for r in rows for w in re.findall(r"\w{6,}", r["input_text"])}
    shared = [w for w in content if boundary_find(b_text, w) != -1]
    print(f"Lexical overlap with Dataset B: {len(shared)}/{len(content)} content words")
