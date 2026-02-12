def calculate_confidence(retrieval_results, max_distance=1.5):
    """
    Convert retrieval distance into confidence score (0 to 1)
    """

    if not retrieval_results:
        return 0.0

    best_distance = retrieval_results[0]["distance"]

    # Distance too high → no confidence
    if best_distance >= max_distance:
        return 0.0

    # Invert distance to confidence
    confidence = 1 - (best_distance / max_distance)

    return round(confidence, 2)
