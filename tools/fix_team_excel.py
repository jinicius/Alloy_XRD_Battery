"""팀 엑셀 03_분석의 지정 칸만 고친다 (시트 XML 직접 수정, 다른 서식 보존)."""
import re, sys, zipfile
from xml.sax.saxutils import escape

SRC, DST = sys.argv[1], sys.argv[2]

NEW = {}
NEW["E9"] = ("4.08→4.80 V c축 수축률은 A 6.94% > B 5.45% > C 4.40%로 작아졌고(01_XRD 피크 기준. "
             "02 시트 c–V 곡선으로 계산하면 7.03 / 5.68 / 4.69%), 300 cycle 용량(108.2 → 158.3 → 181.6 mAh/g)과 "
             "retention(55.5% → 82.0% → 95.0%)은 커졌다. 이에 따라 c축 수축률이 작을수록 300-cycle capacity "
             "retention이 높아 사이클 수명 특성이 우수한, 서로 반대 방향의 경향이 나타났다.")
NEW["D36"] = ("c축–전압 거동(02 시트): A와 B는 4.2 V까지 c축이 증가하다가 감소하고, C는 4.0 V에서 최대가 된 뒤 완만하게 감소한다. "
              "가장 가파른 감소 구간은 H2→H3 상전이에 따른 c축 수축으로 해석되며, 기울기는 A가 4.25–4.30 V에서 약 6.1 Å/V로 가장 크고, "
              "B는 4.20–4.25 V에서 약 2.3 Å/V, C는 4.20 V 이후 약 1.2–1.3 Å/V로 거의 일정하게 감소해 가장 완만하다.\n"
              "물성·성능 비교: 세 시료 중 c축 수축률(4.08→4.80 V, 01_XRD 피크 기준)이 가장 큰 Sample A(6.94%)가 300 cycle 기준 "
              "retention도 가장 낮았다(55.5%). 수축률이 그다음인 Sample B(5.45%)는 retention이 82.0%, 수축률이 가장 작은 "
              "Sample C(4.40%)는 95.0%로, 수축률이 작을수록 retention이 높은 경향이 일관되게 나타났다(02 시트 곡선 기준 수축률 "
              "7.03 / 5.68 / 4.69%로 계산해도 순서는 같다). 다만 시료가 3개뿐이고 입자 크기, 표면 상태, 조성 등 다른 변수가 함께 "
              "작용했을 가능성이 있으므로, 이 결과는 인과관계가 아니라 c축 수축률과 retention이 서로 반대 방향으로 변하는 경향으로 해석한다.")
FIX_SUB = {
    "E7": [("/ C: 18.695°, 19.070°", "/ B: 18.695°, 19.070°")],
    "D35": [("4.08V와 4.50V의 c와 0.1 Å 이내로 대응하므로", "4.08V와 4.50V의 c와 약 0.1 Å(0.11 / 0.10 Å) 차이로 대응하므로")],
    "D37": [("(7.03% → 5.68% → 4.69%)", "(6.94% → 5.45% → 4.40%, 01_XRD 피크 기준)")],
}

with zipfile.ZipFile(SRC) as z:
    files = {n: z.read(n) for n in z.namelist()}

wb = files["xl/workbook.xml"].decode("utf-8")
pfx = re.match(r"<(\w+:)?workbook", wb.lstrip("﻿").split("?>")[-1].strip()).group(1) or ""
rid = re.search(r'<(?:\w+:)?sheet [^>]*name="03_분석"[^>]*r:id="([^"]+)"', wb).group(1)
rels = files["xl/_rels/workbook.xml.rels"].decode("utf-8")
tgt = re.search(rf'<Relationship [^>]*Id="{rid}"[^>]*>', rels).group(0)
tgt = re.search(r'Target="([^"]+)"', tgt).group(1).lstrip("/")
sheet = tgt if tgt.startswith("xl/") else "xl/" + tgt
s = files[sheet].decode("utf-8")
p = re.match(r".*?<(\w+:)?worksheet", s, re.S).group(1) or ""

sst = []
if "xl/sharedStrings.xml" in files:
    ss = files["xl/sharedStrings.xml"].decode("utf-8")
    for si in re.findall(rf"<{p}si>(.*?)</{p}si>", ss, re.S):
        sst.append("".join(re.findall(rf"<{p}t[^>]*>(.*?)</{p}t>", si, re.S)))

from html import unescape


def cell_text(ref):
    m = re.search(rf'<{p}c r="{ref}"([^>]*)>(.*?)</{p}c>', s, re.S)
    attrs, inner = m.group(1), m.group(2)
    if 't="s"' in attrs:
        return unescape(sst[int(re.search(rf"<{p}v>(\d+)</{p}v>", inner).group(1))])
    if 't="inlineStr"' in attrs:
        return unescape("".join(re.findall(rf"<{p}t[^>]*>(.*?)</{p}t>", inner, re.S)))
    raise ValueError(ref)


def set_text(ref, text):
    global s
    m = re.search(rf'<{p}c r="{ref}"([^>]*)>(.*?)</{p}c>', s, re.S)
    style = re.search(r's="(\d+)"', m.group(1))
    st = f' s="{style.group(1)}"' if style else ""
    new = f'<{p}c r="{ref}"{st} t="inlineStr"><{p}is><{p}t xml:space="preserve">{escape(text)}</{p}t></{p}is></{p}c>'
    s = s[:m.start()] + new + s[m.end():]


for ref, subs in FIX_SUB.items():
    t = cell_text(ref)
    for a, b in subs:
        assert a in t, (ref, a)
        t = t.replace(a, b)
    set_text(ref, t)
for ref, t in NEW.items():
    cell_text(ref)
    set_text(ref, t)
files[sheet] = s.encode("utf-8")
with zipfile.ZipFile(DST, "w", zipfile.ZIP_DEFLATED) as z:
    for n, b in files.items():
        z.writestr(n, b)
print("saved", DST, "sheet", sheet)
