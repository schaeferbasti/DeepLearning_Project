from torchmetrics.text.bleu import BLEUScore

predicted_text = ['le' 'centre' 'des' 'editeurs' 'qui' 'chers' 'sen' 'remplacer' 'na' 'pas' 'la' 'conduite' 'et' 'le' 'conseil' 'ougandais' 'de' 'representer' 'est' 'connu' 'pour' 'prendre' 'des' 'decisions' 'disrael' 'qui' 'perpetre' 'le' 'ameliorer' 'de' 'fonctionne' 'de' 'capital' 'radio' 'pour' 'des' 'cesars' 'kenyaneen' 'quil' 'na' 'pas' 'soulever' 'et' 'bien' 'sur' 'travailler' 'avec' 'le' 'centre' 'des' 'uniformes' 'pour' 'la' 'regner' 'de' 'icommons' 'special']
ground_truth_text = ['le' 'centre' 'des' 'media' 'qui' 'devait' 'sen' 'occuper' 'na' 'pas' 'la' 'credibilite' 'et' 'le' 'conseil' 'ougandais' 'de' 'radiodiffusion' 'est' 'connu' 'pour' 'prendre' 'des' 'decisions' 'laxistes' 'qui' 'incluent' 'le' 'renvoi' 'de' 'gaetano' 'de' 'capital' 'radio' 'pour' 'des' 'commentaires' 'homosexuels' 'quil' 'na' 'pas' 'emis' 'et' 'bien' 'sur' 'travailler' 'avec' 'le' 'centre' 'des' 'media' 'pour' 'la' 'deportation' 'de' 'blake' 'lambert']
predicted_text_string = ''.join(predicted_text)
ground_truth_text_string = ''.join(ground_truth_text)

n_grams = [1, 2, 3, 4]
# BLEU metric
for i in n_grams:
    bleu = BLEUScore(n_gram=i, smooth=True)
    bleu_score = bleu(predicted_text_string, ground_truth_text_string)
    print("BLEU " + str(i) + " Score:", bleu_score.item())
    bleu_score = bleu(predicted_text, ground_truth_text)
    print("BLEU " + str(i) + " Score:", bleu_score.item())
    bleu = BLEUScore(n_gram=i, smooth=False)
    bleu_score = bleu(predicted_text_string, ground_truth_text_string)
    print("BLEU " + str(i) + " Score:", bleu_score.item())
    bleu_score = bleu(predicted_text, ground_truth_text)
    print("BLEU " + str(i) + " Score:", bleu_score.item())
    print()