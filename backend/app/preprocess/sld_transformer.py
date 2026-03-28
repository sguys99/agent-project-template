"""KEPCO 154kV 변전소 SLD 이미지 → 구조화 JSON 변환기.

SLD(Single Line Diagram) 이미지를 Vision LLM 1회 호출로
구조화된 JSON 데이터로 변환한다.
"""

import base64
import json
import re
from pathlib import Path
from typing import Optional

import anthropic
from pydantic import BaseModel, Field

# ── 프롬프트 ──────────────────────────────────────────────

SYSTEM_PROMPT = """\
당신은 한국전력(KEPCO) 154kV 변전소 단선결선도(SLD) 분석 전문가입니다.
SLD 이미지를 분석하여 모든 기기 정보와 연결 관계를 JSON으로 추출합니다.

용어 사전:
- T/L = Transmission Line (송전선로)
- M.Tr = Main Transformer (주변압기)
- CB = Circuit Breaker (차단기), DS = Disconnect Switch (단로기)
- LDS = Line Disconnect Switch (선로개폐기)
- OLTC = On-Load Tap Changer (부하시탭절환기)
- S.Tr = Station Transformer (소내변압기)
- 86B = Bus Protection Relay (모선 보호 계전기)

상태 판독 규칙:
- 초록색 원/사각형 = closed (투입)
- 빨간색 원/사각형 = open (개방)

규칙:
- 이미지에 보이지 않는 정보를 추측하지 마세요
- 읽을 수 없는 텍스트는 null로 표시하세요
- 반드시 지정된 JSON 스키마에 맞게 출력하세요"""

USER_PROMPT = """\
이 SLD 이미지를 분석하여 아래 JSON 스키마에 맞게 모든 기기 정보를 추출하세요.

출력 JSON 구조:
{
  "substation": { "name", "voltage_high_kv", "voltage_low_kv", "bank_count", "source_image" },
  "high_voltage_side": {
    "buses": [{ "bus_id", "name", "voltage_kv", "status" }],
    "bus_section": { "bus_tie": { "cb_id", "status", "connects" }, "bus_couplers": [...] },
    "protection": { "86B1": {...}, "86B2": {...} },
    "transmission_lines": [{
      "tl_id", "name", "bus_connection", "power_mw",
      "status_indicators": { "43RC", "43CA", "43PDA_AM" },
      "bus_selector": { "BUS_1": bool, "BUS_2": bool },
      "equipment": { "cb": { "id", "status" }, "ds": [...], "lds": null|{...} }
    }]
  },
  "main_transformers": [{
    "tr_id", "name", "power_mw", "voltage_low_kv", "temperature_c", "tap_position",
    "oltc": { "mode", "remote_local" },
    "protection": { "43PDA_AM", "51SH", "88T" },
    "bus_assignment": { "BUS_1": bool, "BUS_2": bool },
    "high_side_equipment": { "cb", "ds" },
    "low_side_equipment": { "cb" },
    "connected_low_buses": [...]
  }],
  "low_voltage_side": {
    "buses": [{ "bus_id", "name", "status", "fed_by" }],
    "bus_ties": [{ "name", "status", "connects" }],
    "bus_sections": [{ "cb_id", "status", "bus" }],
    "feeders": [{ "feeder_id", "name", "feeder_number", "bus", "cb_ids" }],
    "special_transformers": [{ "name", "feeder_number", "bus", "cb_ids" }]
  }
}

반드시 ```json 블록으로 응답하세요."""


# ── Pydantic 검증 모델 ───────────────────────────────────

class Equipment(BaseModel):
    cb: dict
    ds: list[dict] = Field(default_factory=list)
    lds: Optional[dict] = None


class TransmissionLine(BaseModel):
    tl_id: str
    name: str
    bus_connection: str
    power_mw: Optional[float] = None
    status_indicators: Optional[dict] = None
    bus_selector: Optional[dict] = None
    equipment: dict


class MainTransformer(BaseModel):
    tr_id: str
    name: str
    power_mw: Optional[float] = None
    voltage_low_kv: Optional[float] = None
    temperature_c: Optional[float] = None
    tap_position: Optional[int] = None
    oltc: Optional[dict] = None
    protection: Optional[dict] = None
    bus_assignment: Optional[dict] = None
    high_side_equipment: dict
    low_side_equipment: dict
    connected_low_buses: list[str] = Field(default_factory=list)


class HighVoltageSide(BaseModel):
    buses: list[dict]
    bus_section: Optional[dict] = None
    protection: Optional[dict] = None
    transmission_lines: list[dict]


class LowVoltageSide(BaseModel):
    buses: list[dict]
    bus_ties: list[dict] = Field(default_factory=list)
    bus_sections: list[dict] = Field(default_factory=list)
    feeders: list[dict] = Field(default_factory=list)
    special_transformers: list[dict] = Field(default_factory=list)


class SubstationSLD(BaseModel):
    substation: dict
    high_voltage_side: HighVoltageSide
    main_transformers: list[dict]
    low_voltage_side: LowVoltageSide
    topology_graph: Optional[dict] = None


# ── 메인 함수 ────────────────────────────────────────────

def transform_sld(
    image_path: str,
    model: str = "claude-sonnet-4-5-20250929",
) -> dict:
    """SLD 이미지를 구조화 JSON으로 변환한다.

    Args:
        image_path: SLD 이미지 파일 경로 (JPG)
        model: Vision LLM 모델명

    Returns:
        변전소 SLD 구조화 데이터 (dict)

    Raises:
        FileNotFoundError: 이미지 파일이 없을 때
        json.JSONDecodeError: LLM 응답이 유효한 JSON이 아닐 때
        ValueError: Pydantic 검증 실패 시
    """
    # 1. 이미지 로드
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"SLD 이미지를 찾을 수 없습니다: {image_path}")

    image_bytes = path.read_bytes()
    b64_image = base64.standard_b64encode(image_bytes).decode()

    # 미디어 타입 결정
    suffix = path.suffix.lower()
    media_type_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
    media_type = media_type_map.get(suffix, "image/jpeg")

    # 2. Vision LLM 호출
    raw_text = _call_vision_llm(b64_image, media_type, model)

    # 3. JSON 파싱
    result = _parse_json_response(raw_text)

    # 4. 토폴로지 그래프 생성
    result["topology_graph"] = _build_topology(result)

    # 5. Pydantic 검증
    SubstationSLD(**result)

    return result


def _call_vision_llm(b64_image: str, media_type: str, model: str) -> str:
    """Claude Vision API를 호출하여 SLD 분석 결과를 반환한다."""
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=8192,
        temperature=0,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64_image,
                        },
                    },
                    {"type": "text", "text": USER_PROMPT},
                ],
            }
        ],
    )
    return response.content[0].text


def _parse_json_response(raw_text: str) -> dict:
    """LLM 응답에서 JSON 블록을 추출하여 파싱한다."""
    # ```json ... ``` 블록 추출
    match = re.search(r"```json\s*(.*?)\s*```", raw_text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        # 코드 블록이 없으면 전체 텍스트를 JSON으로 시도
        json_str = raw_text.strip()

    return json.loads(json_str)


def _build_topology(sld: dict) -> dict:
    """추출된 SLD 데이터에서 토폴로지 그래프를 자동 생성한다."""
    nodes: list[dict] = []
    edges: list[dict] = []

    # 154kV 모선
    for bus in sld["high_voltage_side"]["buses"]:
        nodes.append({"id": bus["bus_id"], "type": "bus", "voltage": 154})

    # T/L
    for tl in sld["high_voltage_side"]["transmission_lines"]:
        nodes.append({"id": tl["tl_id"], "type": "transmission_line", "name": tl["name"]})
        edges.append({
            "from": tl["bus_connection"],
            "to": tl["tl_id"],
            "via_cb": tl["equipment"]["cb"]["id"],
            "type": "hv_tl",
        })

    # Bus Tie (154kV)
    bus_section = sld["high_voltage_side"].get("bus_section")
    if bus_section and bus_section.get("bus_tie"):
        bt = bus_section["bus_tie"]
        if bt.get("connects") and len(bt["connects"]) >= 2:
            edges.append({
                "from": bt["connects"][0],
                "to": bt["connects"][1],
                "via_cb": bt["cb_id"],
                "type": "bus_tie",
            })

    # M.Tr
    hv_buses = sld["high_voltage_side"]["buses"]
    for mtr in sld["main_transformers"]:
        nodes.append({"id": mtr["tr_id"], "type": "main_transformer", "name": mtr["name"]})

        # 154kV 측 연결: bus_assignment 기반으로 연결된 모선 찾기
        bus_id = _find_assigned_bus(hv_buses, mtr.get("bus_assignment", {}))
        if bus_id:
            edges.append({
                "from": bus_id,
                "to": mtr["tr_id"],
                "via_cb": mtr["high_side_equipment"]["cb"]["id"],
                "type": "hv_mtr",
            })

        # 23kV 측 연결
        low_cb_id = mtr["low_side_equipment"]["cb"]["id"]
        for low_bus in mtr.get("connected_low_buses", []):
            edges.append({
                "from": mtr["tr_id"],
                "to": low_bus,
                "via_cb": low_cb_id,
                "type": "lv_mtr",
            })

    # 23kV 모선
    for bus in sld["low_voltage_side"]["buses"]:
        nodes.append({"id": bus["bus_id"], "type": "bus", "voltage": 23})

    # Bus Tie (23kV)
    for bt in sld["low_voltage_side"].get("bus_ties", []):
        if bt.get("connects") and len(bt["connects"]) >= 2:
            edges.append({
                "from": bt["connects"][0],
                "to": bt["connects"][1],
                "via_cb": bt["name"],
                "type": "bus_tie",
            })

    return {"nodes": nodes, "edges": edges}


def _find_assigned_bus(hv_buses: list[dict], bus_assignment: dict) -> Optional[str]:
    """bus_assignment에서 True인 모선의 bus_id를 반환한다."""
    for bus in hv_buses:
        bid = bus["bus_id"]
        if "BUS_1" in bid and bus_assignment.get("BUS_1"):
            return bid
        if "BUS_2" in bid and bus_assignment.get("BUS_2"):
            return bid
    # fallback: 첫 번째 True인 키에 대응하는 모선
    for key, val in bus_assignment.items():
        if val:
            for bus in hv_buses:
                if key.replace("BUS_", "") in bus["bus_id"]:
                    return bus["bus_id"]
    return hv_buses[0]["bus_id"] if hv_buses else None


# ── 계통도 텍스트 출력 ──────────────────────────────────────


def _status_icon(status: Optional[str]) -> str:
    """기기 상태를 아이콘으로 변환한다."""
    if status == "closed":
        return "●"
    if status == "open":
        return "○"
    return "?"


def print_sld(sld: dict) -> str:
    """SubstationSLD dict를 텍스트 계통도로 출력한다.

    Args:
        sld: transform_sld() 반환값 (SubstationSLD 형식 dict)

    Returns:
        텍스트 계통도 문자열
    """
    lines: list[str] = []
    W = 80  # 출력 폭

    # ── 헤더 ──
    sub = sld["substation"]
    title = (
        f"{sub['name']} 변전소 단선결선도 "
        f"({sub.get('voltage_high_kv', '?')}/{sub.get('voltage_low_kv', '?')}kV, "
        f"{sub.get('bank_count', '?')}Bank)"
    )
    lines.append("=" * W)
    lines.append(title.center(W))
    lines.append("=" * W)

    # ── 154kV 측 ──
    hv = sld["high_voltage_side"]
    lines.append("")
    lines.append(f"── 154kV 측 {'─' * (W - 13)}")

    # 모선별 연결된 T/L, M.Tr 인덱스 구축
    tl_by_bus: dict[str, list[dict]] = {}
    for tl in hv["transmission_lines"]:
        tl_by_bus.setdefault(tl["bus_connection"], []).append(tl)

    mtr_by_bus: dict[str, list[dict]] = {}
    for mtr in sld["main_transformers"]:
        bus_id = _find_assigned_bus(hv["buses"], mtr.get("bus_assignment", {}))
        if bus_id:
            mtr_by_bus.setdefault(bus_id, []).append(mtr)

    for i, bus in enumerate(hv["buses"]):
        lines.append("")
        voltage_str = f"{bus.get('voltage_kv', '?')}kV" if bus.get("voltage_kv") else "?kV"
        bus_header = f"  [{bus['name']}] {'━' * 40} ({voltage_str}, {bus.get('status', '?')})"
        lines.append(bus_header)

        items: list[str] = []
        # T/L
        for tl in tl_by_bus.get(bus["bus_id"], []):
            cb = tl["equipment"]["cb"]
            entry = f"{tl['tl_id']:<5} {tl['name']:<20} CB:{cb['id']}[{_status_icon(cb.get('status'))}]"
            if tl.get("power_mw") is not None:
                entry += f"  {tl['power_mw']}MW"
            lds = tl["equipment"].get("lds")
            if lds:
                entry += f"  LDS:{lds['id']}[{_status_icon(lds.get('status'))}]"
            items.append(entry)
        # M.Tr
        for mtr in mtr_by_bus.get(bus["bus_id"], []):
            cb = mtr["high_side_equipment"]["cb"]
            entry = f"{mtr['tr_id']:<5} {mtr['name']:<20} CB:{cb['id']}[{_status_icon(cb.get('status'))}]"
            items.append(entry)

        for j, item in enumerate(items):
            prefix = "    └─ " if j == len(items) - 1 else "    ├─ "
            lines.append(prefix + item)

        # Bus Tie (154kV 모선 사이)
        if i < len(hv["buses"]) - 1:
            bus_section = hv.get("bus_section")
            if bus_section and bus_section.get("bus_tie"):
                bt = bus_section["bus_tie"]
                lines.append("")
                lines.append(
                    f"          Bus Tie CB:{bt['cb_id']} [{_status_icon(bt.get('status'))}]"
                )

    # ── 주변압기 ──
    lines.append("")
    lines.append(f"── 주변압기 {'─' * (W - 12)}")

    for mtr in sld["main_transformers"]:
        lines.append("")
        header_parts = [f"  {mtr['name']} ({mtr['tr_id']})"]
        if mtr.get("power_mw") is not None:
            header_parts.append(f"{mtr['power_mw']}MW")
        if mtr.get("voltage_low_kv") is not None:
            header_parts.append(f"{mtr['voltage_low_kv']}kV")
        if mtr.get("temperature_c") is not None:
            header_parts.append(f"{mtr['temperature_c']}°C")
        if mtr.get("tap_position") is not None:
            header_parts.append(f"TAP:{mtr['tap_position']}")
        oltc = mtr.get("oltc")
        if oltc:
            header_parts.append(f"OLTC:{oltc.get('mode', '?')}/{oltc.get('remote_local', '?')}")
        lines.append("  ".join(header_parts))

        # 154kV 측 장비
        hs = mtr["high_side_equipment"]
        hs_cb = hs["cb"]
        hs_line = f"    154kV ← CB:{hs_cb['id']}[{_status_icon(hs_cb.get('status'))}]"
        ds_list = hs.get("ds", [])
        if ds_list:
            ds_strs = [f"{d['id']}[{_status_icon(d.get('status'))}]" for d in ds_list]
            hs_line += f"  DS:{','.join(ds_strs)}"
        lines.append(hs_line)

        # 23kV 측 장비
        ls_cb = mtr["low_side_equipment"]["cb"]
        low_buses = mtr.get("connected_low_buses", [])
        ls_line = f"    23kV  → CB:{ls_cb['id']}[{_status_icon(ls_cb.get('status'))}]"
        if low_buses:
            # bus_id에서 간결한 이름 추출
            bus_names = [b.replace("23kV_", "#") for b in low_buses]
            ls_line += f"  → {', '.join(bus_names)}"
        lines.append(ls_line)

    # ── 23kV 측 ──
    lv = sld["low_voltage_side"]
    lines.append("")
    lines.append(f"── 23kV 측 {'─' * (W - 12)}")

    # 피더/S.Tr을 모선별로 그룹화
    feeders_by_bus: dict[str, list[dict]] = {}
    for fdr in lv.get("feeders", []):
        feeders_by_bus.setdefault(fdr["bus"], []).append(fdr)

    str_by_bus: dict[str, list[dict]] = {}
    for st in lv.get("special_transformers", []):
        str_by_bus.setdefault(st["bus"], []).append(st)

    # Bus Tie 인덱스: 두 모선 중 먼저 나오는 모선 뒤에 표시
    bus_ids = [b["bus_id"] for b in lv["buses"]]
    bus_ties_after: dict[str, list[dict]] = {}
    for bt in lv.get("bus_ties", []):
        connects = bt.get("connects", [])
        if len(connects) >= 2:
            # 두 모선 중 목록에서 먼저 나오는 모선 뒤에 배치
            idx0 = bus_ids.index(connects[0]) if connects[0] in bus_ids else 999
            idx1 = bus_ids.index(connects[1]) if connects[1] in bus_ids else 999
            first_bus = connects[0] if idx0 < idx1 else connects[1]
            bus_ties_after.setdefault(first_bus, []).append(bt)

    for bus in lv["buses"]:
        lines.append("")
        fed_by = bus.get("fed_by", "?")
        bus_header = f"  [{bus['name']}] {'━' * 20} ({bus.get('status', '?')}, fed by {fed_by})"
        lines.append(bus_header)

        items: list[str] = []
        for fdr in feeders_by_bus.get(bus["bus_id"], []):
            cb_str = ",".join(fdr.get("cb_ids", []))
            items.append(f"F{fdr.get('feeder_number', '?'):<4} {fdr['name']:<6} CB:{cb_str}")
        for st in str_by_bus.get(bus["bus_id"], []):
            cb_str = ",".join(st.get("cb_ids", []))
            items.append(f"{st['name']:<10} CB:{cb_str}")

        for j, item in enumerate(items):
            prefix = "    └─ " if j == len(items) - 1 else "    ├─ "
            lines.append(prefix + item)

        # Bus Tie
        for bt in bus_ties_after.get(bus["bus_id"], []):
            lines.append("")
            lines.append(
                f"          Bus Tie {bt['name']} [{_status_icon(bt.get('status'))}]"
            )

    lines.append("")
    lines.append("=" * W)

    text = "\n".join(lines)
    print(text)
    return text
