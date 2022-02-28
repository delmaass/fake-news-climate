import spacy
import unicodedata

TOPIC_LEMMA = ["rechauffement", "climat", "GIEC", "COP", "ONU", "dereglement", "co2", "emission"]

def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])

def extract_topic_clauses(doc):
    seen = set() # keep track of covered words
    chunks = []

    for sentence in doc.sents:
        heads = [cc for cc in sentence.root.children if cc.dep_ == 'conj']

        for head in heads:
            words = [ww for ww in head.subtree]
            for word in words:
                seen.add(word)
            chunk = (' '.join([ww.text for ww in words]))
            chunks.append( (head.i, chunk) )

        unseen = [ww for ww in sentence if ww not in seen]
        chunk = ' '.join([ww.text for ww in unseen])
        chunks.append( (sentence.root.i, chunk) )
        
        
        #         displacy.serve(sentence, style="dep")

    chunks = sorted(chunks, key=lambda x: x[0])
    remaining_chunks = []

    for _, chunk in chunks:
        clause = nlp(chunk)
        clause_lemma = [token.lemma_ for token in clause]
        clause_lemmatized = remove_accents(' '.join(clause_lemma))

        for topic_lemma in TOPIC_LEMMA:
            if topic_lemma in clause_lemmatized:
                remaining_chunks.append(clause_lemma)
                break

    return remaining_chunks

if __name__ == "__main__":
    nlp = spacy.load("fr_core_news_md")

    sample_text = "Selon lui, cette similitude permettait de conclure que l’évolution du CO2 atmosphérique pilotait celle de la température (et donc que nos émissions massives de CO2 allaient réchauffer la Terre de façon catastrophique). En réalité, on sait depuis longtemps, et c’est même l’un des arguments les plus solides du climato-réalisme, que les deux courbes sont légèrement décalées l’une par rapport à l’autre, dans « le mauvais sens » pour les alarmistes : la température précède le CO2, d’environ 800 ans en moyenne. Comme un effet ne peut pas précéder sa cause, la conclusion est claire : le CO2 n’est pas le moteur de l’évolution de la température. C’est le contraire qui est vrai : l’évolution de la température entraîne celle du CO2, avec un retard de quelques centaines d’années qui correspond à une durée appelée « mémoire thermique » des océans. (Lorsque la température monte, les océans dégazent plus facilement le CO2 qu’ils contiennent, mais le processus démare avec lenteur.) Cala implique que nous n’avons pas de raison particulière de craindre un effet de nos émissions de gaz carbonique sur la température globale. Pascal Richet, comme bien d’autres, ne manque pas de revenir sur cette évidence, frappée au coin du bon sens et d’une logique qu’approuvait Aristote aussi bien qu’Occam (celui du rasoir). Le bricolage carbocentriste Mais les carbocentristes n’ont pas laissé ces bêtes questions de cause et de conséquence gâcher leur si belle histoire d’apocalypse climatique provoquée par l’Homme. Face à l’objection, ils ont fait ce que font bien plus de scientifiques qu’on imagine : ils ont bricolé quelque chose pour sauver le soldat CO2."
    sample_doc = nlp(sample_text)

    print(extract_topic_clauses(sample_doc))