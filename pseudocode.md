BPE_TRAIN(corpus, num_merges):
    vocabulary = count_unique_words(corpus)

    for each word in vocabulary:
        symbols[word] = utf8_characters(word) + ["</w>"]

    for rank in 0 .. num_merges - 1:
        pair_counts = empty map

        for each word_type:
            for each adjacent pair in symbols[word_type]:
                pair_counts[pair] += frequency(word_type)

        if pair_counts is empty:
            stop

        best_pair = most_frequent_pair(pair_counts)

        if best_pair.frequency < min_pair_frequency:
            stop

        save best_pair as merge rule with current rank

        for each word_type:
            replace every non-overlapping occurrence of best_pair

    learned_vocabulary = final symbols


BPE_TOKENIZE(word):
    symbols = utf8_characters(word) + ["</w>"]

    for each merge_rule in rank order:
        replace every non-overlapping occurrence of merge_rule.left + merge_rule.right

    remove "</w>" from final symbols

    return symbols


FST_ADD_SUFFIX_RULE(suffix):
    symbols = utf8_characters(suffix)

    state = start_state

    for symbol in reverse(symbols):
        if transition(state, symbol) does not exist:
            create new state
            create transition(state, new_state, symbol)
        state = next_state

    mark state as accepting for this suffix rule


FST_SEGMENT(word):
    symbols = utf8_characters(word)

    dp[n] = valid empty suffix chain

    for end from n down to 1:
        if dp[end] is invalid:
            continue

        matches = all suffix rules ending at position end

        for each match:
            start = match.start_position
            candidate_chain = match.rule + dp[end]
            dp[start] = best(dp[start], candidate_chain)

    choose root boundary i such that:
        dp[i] is valid
        prefix word[0:i] is a valid root
        suffix coverage is maximal

    if no valid boundary exists:
        return [word]

    return [root] + suffixes


FST_BPE_TRAIN(corpus):
    segmented_corpus = []

    for each word in corpus:
        segments = FST_SEGMENT(word)
        append all segments to segmented_corpus

    train BPE on segmented_corpus


FST_BPE_TOKENIZE(word):
    segments = FST_SEGMENT(word)
    output = []

    for each segment:
        output += BPE_TOKENIZE(segment)

    return output
