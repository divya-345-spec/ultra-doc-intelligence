import re


def extract_structured_fields(full_text: str) -> dict:
    """
    Regex-based extraction of the 11 required logistics fields.
    Returns all keys with None for missing values.
    """
    if not isinstance(full_text, str):
        full_text = " ".join([str(x) for x in full_text])

    text = re.sub(r"\s+", " ", full_text)
    lower = text.lower()

    def find(patterns, source=lower):
        for p in patterns:
            m = re.search(p, source, re.IGNORECASE)
            if m:
                return m.group(1).strip()
        return None

    # --- shipment_id ---
    shipment_id = find([
        r"shipment\s*(?:id|#|no|number)\s*[:\-]?\s*([A-Z0-9\-]+)",
        r"load\s*(?:id|#|no|number)\s*[:\-]?\s*([A-Z0-9\-]+)",
        r"order\s*(?:id|#|no|number)\s*[:\-]?\s*([A-Z0-9\-]+)",
        r"pro\s*(?:#|number)\s*[:\-]?\s*([A-Z0-9\-]+)",
        r"bol\s*(?:#|number)\s*[:\-]?\s*([A-Z0-9\-]+)",
        r"\b(ld\d+)\b",
    ], source=text)

    # --- shipper ---
    shipper = find([
        r"shipper\s*[:\-]\s*(.+?)(?:\n|$|consignee|receiver|deliver)",
        r"ship\s*from\s*[:\-]\s*(.+?)(?:\n|$)",
        r"origin\s*[:\-]\s*(.+?)(?:\n|$)",
        r"pickup\s*(?:location|address|from)\s*[:\-]\s*(.+?)(?:\n|$)",
    ])

    # --- consignee ---
    consignee = find([
        r"consignee\s*[:\-]\s*(.+?)(?:\n|$|shipper|deliver)",
        r"receiver\s*[:\-]\s*(.+?)(?:\n|$)",
        r"ship\s*to\s*[:\-]\s*(.+?)(?:\n|$)",
        r"destination\s*[:\-]\s*(.+?)(?:\n|$)",
        r"deliver(?:y)?\s*(?:location|address|to)\s*[:\-]\s*(.+?)(?:\n|$)",
    ])

    # --- pickup_datetime ---
    pickup_datetime = find([
        r"pickup\s*(?:date|time|datetime)\s*[:\-]\s*(.+?)(?:\n|$|delivery|deliver)",
        r"pick\s*up\s*[:\-]\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}(?:\s+\d{1,2}:\d{2})?)",
        r"pickup\s*[:\-]\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}(?:\s+\d{1,2}:\d{2})?)",
    ])

    # --- delivery_datetime ---
    delivery_datetime = find([
        r"delivery\s*(?:date|time|datetime)\s*[:\-]\s*(.+?)(?:\n|$|pickup|pick)",
        r"deliver(?:y)?\s*[:\-]\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}(?:\s+\d{1,2}:\d{2})?)",
        r"drop\s*off\s*[:\-]\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}(?:\s+\d{1,2}:\d{2})?)",
    ])

    # --- equipment_type ---
    equipment_type = find([
        r"equipment\s*(?:type)?\s*[:\-]\s*(.+?)(?:\n|$)",
        r"\b(flatbed|reefer|dry\s*van|van|tanker|intermodal|container)\b",
    ])

    # --- mode ---
    mode = find([
        r"mode\s*[:\-]\s*(\w+)",
        r"\b(ftl|ltl|fcl|lcl|parcel|intermodal|drayage|truckload)\b",
    ])

    # --- rate ---
    rate = None
    currency = None

    rate_match = re.search(
        r"(?:agreed\s*amount|carrier\s*rate|total\s*(?:rate|charge|amount)|rate|charge)\s*[:\-]?\s*"
        r"\$?\s*([\d,]+\.?\d*)",
        lower
    )
    if rate_match:
        rate = rate_match.group(1)

    if not rate:
        dollar_match = re.search(r"\$\s*([\d,]+\.?\d*)", text)
        if dollar_match:
            rate = dollar_match.group(1)

    # --- currency ---
    currency_match = re.search(r"\b(usd|inr|eur|gbp|cad)\b", lower)
    if currency_match:
        currency = currency_match.group(1).upper()
    elif "$" in text:
        currency = "USD"

    # --- weight ---
    weight = find([
        r"weight\s*[:\-]\s*([\d,]+\.?\d*\s*(?:lbs?|kg|tons?)?)",
        r"([\d,]+\.?\d*)\s*(?:lbs?|kg|tons?)\b",
    ])

    # --- carrier_name ---
    carrier_name = find([
        r"carrier\s*(?:name)?\s*[:\-]\s*(.+?)(?:\n|$|driver|dispatcher|rate)",
        r"trucking\s*company\s*[:\-]\s*(.+?)(?:\n|$)",
    ])

    return {
        "shipment_id": shipment_id,
        "shipper": shipper,
        "consignee": consignee,
        "pickup_datetime": pickup_datetime,
        "delivery_datetime": delivery_datetime,
        "equipment_type": equipment_type,
        "mode": mode,
        "rate": rate,
        "currency": currency,
        "weight": weight,
        "carrier_name": carrier_name,
    }
