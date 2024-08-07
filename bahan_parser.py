##import all the required libraries
import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
import simplemma
from nltk.tokenize import word_tokenize


nltk.download('stopwords')
nltk.download('wordnet')

def clean_number(text):
    text = re.sub(r'\w*\d+\w*', '', str(text))
    return text
#casefolding
def token_lower(text):
    text = ''.join(str(text)).lower() # lowercase text
    return text

#Remove Puncutuation
clean_spcl = re.compile('[/(){}\[\]\|@,;]')
clean_symbol = re.compile('[^0-9a-z]')
def clean_punct(text):
    text = clean_spcl.sub('', text)
    text = clean_symbol.sub(' ', text)
    return text

#menghapus double atau lebih whitespace
def _normalize_whitespace(text):
    corrected = str(text)
    corrected = re.sub(r"//t",r"\t", corrected)
    corrected = re.sub(r"( )\1+",r"\1", corrected)
    corrected = re.sub(r"(\n)\1+",r"\1", corrected)
    corrected = re.sub(r"(\r)\1+",r"\1", corrected)
    corrected = re.sub(r"(\t)\1+",r"\1", corrected)
    return corrected.strip(" ")

#stopword
#clean stopwords
stopwords_indonesian = set(stopwords.words('indonesian'))

# Hapus kata "tahu" dari daftar stopwords
stopwords_indonesian.discard("tahu")

takaran = ['gram', 'sdt', 'sdm', 'buah', 'ml', 'mililiter', 'cup', 'gelas', 'tsp', 'tbsp', 'liter', 'g', 'kg',
           'kilogram', 'pound', 'ons', 'pint', 'pt', 'c', 'celcius', 'f', 'fahrenheit', 'derajat', 'm', 'meter',
           'mm', 'milimeter', 'ruas', 'cm', 'centimeter', 'senti', 'centi', 'siung', 'lembar', 'tangkai', 'batang',
           'butir', 'slice', 'keping', 'kotak', 'papan', 'genggam', 'sejumput', 'bungkus', 'sachet', 'dcc', 'piring',
           'cincin', 'grambahan', 'sendok', 'porsi', 'kantung', 'kantong', 'scoop', 'skup', 'scop', 'loyang', 'segenggam',
           'buntil', 'ikat', 'double', 'botol', 'pack', 'siung','diameter', 'ekor', 'kuntum', 'bonggol']

word_to_remove = [ "a", "acting", "adaan", "adon", "aduk", "air", "alas", "aluminium", "ambil", "ampas", "aneka", "anti",
                  "api" "arik", "asap", "ayak", "b", "bagi", "bagus", "bahan", "bakar", "baluran", "bambu", "bantu", "basah", "basic", 
                  "batangsaus", "belah", "beli", "bening", "bentuk", "berat", "bersih", "berseta", "beserta" "biar", "bilas", 
                  "blender", "bola", "buahbahan", "buahsaus", "buahlainnya", "buahkuah", "buahpencelup", "buang", "bulat", 
                  "bunda", "butir", "butirkuah", "bumbu", "cabut", "cacah", "cair", "cairkan", "campur", "celup", "cepat", 
                  "cetak", "cincang", "cocolan", "cuci", "dadu", "dasar", "diam", "dibagi", "dibentuk" "didih",  "digeprek", 
                  "dingin", "diiikat", "dimemarkan", "dipulung", "direbus", "disangrai", "diseduh", "ditorch", "duri", 
                  "ekorbumbu", "empuk", "encer", "encerkan", "endapan", "es", "fillet", "filling", "foil", "formula", "garam", 
                  "garnish", "gepeng", "geprak", "geprek", "gigi", "gramisi", "gramlapisan", "gramtaburan", "gula", "gurih", 
                  "goreng", "goyang", "hancur", "hangat", "hangatkan", "halus", "hasil", "hias", "i", "ii", "iii", "iv", "ikat", 
                  "instan", "iris", "isi", "jala", "jam", "jari", "jaripelengkap", "jenis", "jumput", "k", "kain", "kaku", "kapal", 
                  "kasar", "kasir", "kasur", "kedalam", "kerat", "kerok", "kocok", "kocock", "kondisi", "konsistensi", "korea", 
                  "korek", "kotor", "kuah", "kualitas", "kuku", "kukus", "kulkas", "kupas", "larut", "larutkan", "lauk", "lapis", 
                  "lawan", "lebar", "lebih", "leleh", "lelehkan", "lembarbumbu", "lembarlainnya", "lembarpudding", "lembartaburan", 
                  "lembut", "lengkap", "lengket", "lentur", "lepas", "lidi", "lihat", "lilit", "literlainnya", "luar", "lulur", 
                  "lumat", "lumatkan", "lumur", "lumuran", "makan", "mangkuk", "marinasi", "matang", "mekar", "memar", "memarkan", 
                  "menit", "menggoreng", "mengunkep", "mengungkep", "mentah", "menyemat", "menyerong", "merebus", "mlbahan", "mlisi", 
                  "mlfilling", "mlpelengkap", "mlsaus", "mlsambal", "mltaburan", "mltopping", "minimal", "minyak", "mudah", "oil", 
                  "oles", "olesisian", "opsional", "optional", "orak", "panas", "panggang", "pakai", "pasir", "papanpelapis", "pecah", 
                  "pecel", "pekat", "pembalur", "penuh", "pera", "peras", "perasan", "perasannya", "perlembar", "persannya", "peta", "parut",
                  "petik", "pilih", "pipil", "pisah", "plastik", "potong", "protein", "puter", "putus", "rajang", "rambut", "rasa", 
                  "ready", "rebus", "rebusan" "rekat", "remas", "rendah", "rendam", "resep", "robek", "ruang", "sangrai", "sampai", 
                  "saring", "sebentar", "secukupnya", "secukupnyaes", "secukupnyisi", "secukupnyaisian", "secukupnyatumisan", 
                  "secukupnyalainnya", "secukupnyaperendam", "secukupnyakaldu", "secukupnyabubur", "secukupnyabiang", "secukupnyalapisan", 
                  "sedang", "seduh", "sdtserundeng", "sdtsaus", "sdtbaluran", "sdttopping", "sdtlainnya", "sdtmerebus", "sdtcuko", 
                  "sdtpencelup", "sdtbahan", "sdtgarnish", "sdtsambal", "sdtperendam", "sdttaburan", "sdtolesan", "segar", "seger", 
                  "sejumpuhkuah", "sejumputkulit", "sela", "selam", "semalam", "semalaman", "selera", "serat", "serong", "serut", 
                  "serutan", "sesuai", "setengah", "seujung", "siangi", "sikat", "simpul", "simpukan", "simpulkan", "siram", "sisa", 
                  "sisir", "siungbumbu", "siungpelapis" "sdmisi", "sdmtaburan", "sdmbahan", "sdmcuko", "sdmuleg", "sdmserundeng", "sd", 
                  "sobek", "spread", "suhu", "suir", "suwir", "tabur", "taburan", "tali", "tambah", "tangan", "tahan", "tbmsp", "tebal", 
                  "tengah", "tetes", "tetessaus", "tinggi", "tintanya", "tipis", "tiris", "tlapisan", "to", "topping", "torch", "tua", "tumbuk", 
                  "tumis", "tusuk", "ukur", "ulek", "uleg", "utuh", "variasi", "versi", "wadah", "wajan"
                 ]

def clean_stopwords(text):
    words = text.split()
    cleaned_words = [word for word in words if word.lower() not in stopwords_indonesian and
                     word.lower() not in takaran and word.lower() not in word_to_remove]
    cleaned_text = ' '.join(cleaned_words)
    return cleaned_text

def lemmatization(text):
    text = ' '.join(simplemma.lemmatize(word, lang='id') for word in text.split() if word in text)
    return text 

def tokenize_sentence(sentence):
    sentence = re.sub(r'\[|\]', '', sentence)  # Remove brackets
    multi_word_tokens = [
        "abon ayam", "abon sapi", "agar agar", "asam gelugur", "asam jawa", "asam kandis", "asam sunti" "ati ampela", "ati ampela ayam", "ayam kampung", "ayam jantan", 
        "babat sapi", "baby buncis", "baby cumi", "baby kailan", "baby kol",  "baby potato", "bakso ayam", "bakso sapi", "baking soda", "bawang bombay", 
        "bawang bombai", "bawang merah", "bawang putih", "bawang putih bubuk", "baking powder", "bawang prei", "belimbing sayur", "belimbing wuluh", 
        "beras jagung", "beras ketan putih", "beras basmati", "beras ketan", "biji selasih", "biskuit susu", "biskuit oreo", "biskuit marie regal", "bread crumbs", "bread improver",
        "brown sugar", "buah cherry", "buah naga", "buah naga merah", "bubuk spekuk", "bubuk kari", "bunga cengkeh", "bunga genjer", "bunga kecombrang", "bunga lawang", 
        "bunga kol", "bunga sedap malam", "bunga pepaya", "buntut sapi", "butter cream", "cabai merah", "cabai merah besar", "cabai merah keriting", 
        "cabai rawit", "cabai rawit merah", "cabai rawit hijau",  "cabai bubuk", "cabai hijau", "cabai hijau besar", "cake emulsifier", "cabe keriting", 
        "cabai keriting", "cakalang fufu", "ceker ayam", "cheese spread", "chia seed", "choco chips", "coklat bubuk", "cokelat bubuk", "coklat putih",  "chicken katsu", 
        "corn syrup", "cooking cream", "crab stick", "cream cheese", "cream of tartar", "cumi cumi", "daging ayam", "dada ayam", "daging giling", "daging ikan", "daging kambing", "daging sapi", 
        "daging sapi sandung lamur", "daging sapi has dalam", "daun bawang", "daun bayam", "daun cengkeh", "daun cincau", "daun bawang", "daun jeruk", "daun jeruk purut", "daun kari", "daun katuk", 
        "daun kemangi", "daun kol", "daun kunyit", "daun ketumbar", "daun melinjo", "daun mint", "daun pala", "daun pandan", "daun pepaya", "daun pisang", "daun salam", "daun selada", "daun seledri", 
        "daun serai" "daun singkong", "english muffin", "es batu", "essence vanilla", "french fries", "gabin tawar", "gula aren", "gula bubuk", "gula merah", "green tea", "hati sapi", "ice cream", "iga kambing", 
        "iga sapi", "ikan nila", "ikan kembung", "ikan mujair", "ikan mas", "ikan bandeng", "ikan selar", "ikan teri", "ikan gabus", "ikan patin", "ikan tuna", "ikan tongkol", "ikan cakalang", "ikan salmon", 
        "ikan salai", "ikan lele", "ikan gurame", "ikan kakap", "ikan tenggiri", "ikan bawal", "ikan kerapu", "ikan peda", "ikan salem", "ikan asin", "ikan dori", 
        "ikan pe", "jambal roti", "jamur kancing", "jamur kuping",  "jamur hioko", "jamur enoki", "jamur shitake", "jamur champignon", "jamur tiram", "jamur merang", "jambu air", 
        "jantung pisang", "jeruk nipis", "jeruk limau", "jeroan kambing", "jeruk kasturi", "jeruk purut", "jambu air", "jambu biji merah",  "jambu biji", "jeroan sapi", "kacang tanah", 
        "kayu manis",  "kaldu ayam",  "kaldu ayam bubuk", "kaldu bubuk", "kaldu jamur", "kaldu jamur bubuk",  "kaldu sapi", "kaldu udang", "kacang almond", "kacang hijau", "kacang merah", 
        "kacang mede", "kacang kapri", "kacang kedelai", "kacang kenari", "kacang tanah", "kacang panjang", "kacang polong", "kaki kambing", "kaki sapi", "kapur sirih", "kayu manis", "kerang dara", 
        "kembang tahu", "kembang turi", "kecap manis", "kecap ikan", "ketan hitam", "kelapa parut",  "kecap inggris", "kecap asin", "keju cheddar", "keju edam", "keju parmesan", 
        "keju slice", "keju spread", "keju mozarella", "kerang hijau", "kerupuk kanji", "kerupuk krecek", "kerupuk kulit", "kerupuk merah", "kerupuk udang", "kelapa muda", "kembang turi", "kental manis", 
        "kentang goreng", "ketumbar bubuk", "kikil sapi", "kulit pangsit", "kulit lumpia", "kuning telur", "kulit melinjo", "kulit pastry", "kol putih", "kol ungu", "kol merah", "kolang kaling", "krim whipping", 
        "krimer kental manis", "labu kuning", "labu siam", "lada bubuk", "lemak sapi", "lemon cui", "lidah buaya", "lidah sapi", "lobak putih", "mangga kweni", "matcha powder", "melinjo merah", "melinjo hijau", 
        "merica bubuk", "mie kuning", "nasi putih", "nasi shirataki", "nata de coco", "paha ayam", "pala bubuk", "palm sugar", "paprika hijau", "paprika kuning", "paprika merah", "pasta coklat", "pasta mocca", "paru sapi", "petis udang", "pindang tongkol", "pisang ambon", 
        "pisang candi", "pisang kapok", "pisang kepok", "pisang klutuk", "pisang lilin", "pisang mas", "pisang raja", "pisang tanduk", "pisang uli", "putih telur", "quaker oat", "ragi instant", "rice noodles", "roti burger", "roti john", 
        "roti tawar", "rumput laut", "santan instan", "sayap ayam", "sagu mutiara", "sambal terasi", "sandung lamur", "saus sambal", "saus tartar", "saus tomat", "saus tiram", "sawi putih", "selai coklat", 
        "selai coklat", "selai strawberry", "selai kacang", "selai nanas", "smoked beef", "soda kue", "soft cream", "sosis ayam", "sosis sapi", "star anise", "susu bubuk", "susu cair", "susu evaporasi", 
        "susu full cream", "susu kedelai", "susu kental manis", "susu uht", "susu sapi", "tahu cina", "tahu coklat", "tahu gembos", "tahu jepang", "tahu kuning", "tahu kulit", "tahu putih", 
        "tahu pong", "tahu sutra", "tahu sumedang", "tape ketan", "tape singkong", "tape ketan hitam", "tetelan sapi", "telur asin", "telur ayam", "telur bebek", "telur ikan", 
        "telur puyuh", "temu kunci", "tepung beras", "tepung hunkwe", "teri nasi", "tepung kanji", "tepung ketan", "tepung ketan putih", "tepung panir", "tepung roti", "tepung sagu", 
        "tepung tapioka", "tepung terigu", "tepung maizena", "teri nasi medan", "teri medan", "timun suri", "timun jepang" "tomat merah", "tomat hijau", "tulang ayam", "tulang daging", "tulang kambing", 
        "tulang kaki sapi", "tulang jambal roti", "tulang rangu", "ubi merah", "ubi ungu", "ubi cilembu", "ubi jalar", "ubi kuning", "ubi orange", "unsalted butter", "usus ayam", "urat sapi", "whipping cream", "wijen sangrai"
    ]
    
    # Replace multi-word tokens with versions that have spaces
    for token in multi_word_tokens:
        space_removed_token = token.replace(" ", "")
        sentence = sentence.replace(space_removed_token, token)

    # Tokenize the sentence based on spaces
    words = sentence.split()

    # Create tokens by combining two and three consecutive words
    combined_tokens = []
    i = 0
    while i < len(words):
        two_word_token = " ".join(words[i:i+2])
        three_word_token = " ".join(words[i:i+3])

        if three_word_token in multi_word_tokens:
            combined_tokens.append(three_word_token)
            i += 3
        elif two_word_token in multi_word_tokens:
            combined_tokens.append(two_word_token)
            i += 2
        else:
            combined_tokens.append(words[i])
            i += 1

    # Remove duplicates
    unique_tokens = list(set(combined_tokens))

    return unique_tokens

def bahan_parser(text):
    synonym_mapping = {
        'kubis': 'kol',
        'daun cengkeh': 'ceengkeh',
        'baby kol': 'cuciwis',
        'pekak': 'bunga lawang',
        'kembang lawang': 'bunga lawang',
        'kembang kol': 'bunga kol',
        'bawang bombay': 'bawang bombai',
        'terigu': 'tepung terigu',
        'maizena': 'tepung maizena',
        'tapioka': 'tepung tapioka',
        'kental manis': 'susu kental manis',
        'coklat': 'cokelat',
        'mete': 'kacang mete',
        'almond': 'kacang almond',
        'daun kol': 'kol',
        'daging ayam': 'ayam',
        'selasih': 'biji selasih',
        'agar': 'agar agar',
        'cherry': 'buah cherry',
        'belimbing sayur': 'belimbing wuluh',
        'santan instan': 'santan',
        'star anise': 'bunga lawang',
        'pete': 'petai',
        'bawang prei': 'daun bawang',
        'whipcream': 'whipping cream',
        'laos': 'lengkuas',
        'jeruk kasturi': 'lemon cui',
        'tahu sutra': 'tahu jepang',
        'whip cream': 'whipping cream',
        'daun bayam': 'bayam',
        'pindang tongkol': 'ikan tongkol',
        'tongkol': 'ikan tongkol',
        'sereh': 'serai',
        'daun serai': 'serai',
        'tomat merah': 'tomat',
        'kol ungu': 'kol merah',
        'kemangi': 'daun kemangi',
        'tauge': 'toge',
        'bandeng': 'ikan bandeng',
        'bread crumbs': 'breadcrumbs',
        'bread crumbs': 'tepung roti',
        'es batu': 'es',
        'cakalang fufu': 'ikan cakalang',
        'kyuri': 'timun jepang',
        'ceker': 'ceker ayam',
        'nenas': 'nanas',
        'kacang kapri': 'kapri',
        'daging': 'daging sapi',
        'petai': 'pete'
    }

    text = clean_number(text)
    text = token_lower(text)
    text = clean_punct(text)
    text = _normalize_whitespace(text)
    text = lemmatization(text)
    text = clean_stopwords(text)
    
    # Use the modified tokenize_sentence function
    text = tokenize_sentence(text)

    # Replace synonyms using the synonym mapping
    text = [synonym_mapping.get(word, word) for word in text]

    return text

if __name__ == "__main__":
    recipe_df = pd.read_csv("data/resep_dataset2.csv")
    recipe_df["bahan_parsed"] = recipe_df["Bahan"].apply(
        lambda x: bahan_parser(x)
    )
    df = recipe_df[["Judul", "bahan_parsed", "Bahan", "Step"]]
    df = recipe_df.dropna()
    df.to_csv("data/train_clean.csv", index=False)