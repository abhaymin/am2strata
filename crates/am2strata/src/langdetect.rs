//! The language of a text, for speaking it in that language (`/v1/audio/speech` with `"language": "auto"`).
//!
//! Small and local: no model, no download. The script decides first (Devanagari is Hindi, Hangul Korean, and so
//! on); text in Latin letters is matched against each language's most common short words, and a few letters only
//! one language uses. Short or unclear text is English. It returns ISO 639-1 codes, also for languages no speech
//! model here speaks yet (Chinese, Japanese, Hebrew, Thai), so the caller can say so instead of mispronouncing them.

/// The most common words of each language written in Latin letters (lower case). Words shared by several languages
/// count for each of them; the language with the most matches wins.
const WORDS: &[(&str, &[&str])] = &[
    ("en", &["the", "and", "is", "are", "of", "to", "in", "that", "it", "you", "this", "with", "for", "was", "have", "not", "what", "how", "can", "will"]),
    ("es", &["este", "esta", "tu", "su", "muy", "hola", "gracias", "sí", "qué", "porque", "también", "tengo", "el", "la", "los", "las", "de", "que", "y", "es", "en", "un", "una", "por", "para", "con", "no", "lo", "como", "pero", "más", "está"]),
    ("fr", &["bonjour", "merci", "oui", "très", "mais", "votre", "je", "il", "elle", "sont", "le", "la", "les", "des", "de", "et", "est", "un", "une", "que", "pour", "pas", "dans", "avec", "vous", "nous", "ce", "qui", "sur", "c'est"]),
    ("de", &["hallo", "danke", "ja", "nein", "sehr", "aber", "dein", "mein", "wir", "haben", "der", "die", "das", "und", "ist", "nicht", "ein", "eine", "ich", "zu", "mit", "den", "von", "sie", "es", "auf", "für", "sind", "auch", "wie"]),
    ("it", &["ciao", "grazie", "sì", "molto", "perché", "questa", "tuo", "mio", "ho", "hai", "il", "lo", "la", "gli", "di", "che", "e", "è", "un", "una", "per", "non", "con", "sono", "del", "della", "come", "ma", "questo", "anche"]),
    ("pt", &["olá", "obrigado", "obrigada", "sim", "muito", "seu", "sua", "também", "porque", "tenho", "o", "a", "os", "as", "de", "que", "e", "é", "um", "uma", "não", "para", "com", "do", "da", "em", "você", "mais", "como", "está"]),
    ("nl", &["hallo", "dank", "ja", "nee", "heel", "jouw", "mijn", "wij", "hebben", "dit", "de", "het", "een", "en", "is", "van", "dat", "niet", "ik", "je", "op", "te", "met", "zijn", "voor", "maar", "ook", "wat", "hoe", "er"]),
    ("sv", &["och", "att", "det", "är", "en", "ett", "jag", "inte", "på", "som", "för", "med", "har", "till", "av", "den", "du", "vi", "kan", "om"]),
    ("da", &["og", "at", "det", "er", "en", "et", "jeg", "ikke", "på", "som", "for", "med", "har", "til", "af", "den", "du", "vi", "kan", "hvad"]),
    ("no", &["og", "at", "det", "er", "en", "et", "jeg", "ikke", "på", "som", "for", "med", "har", "til", "av", "den", "du", "vi", "kan", "hva"]),
    ("fi", &["ja", "on", "ei", "se", "että", "hän", "mutta", "kuin", "ole", "olen", "tämä", "mitä", "voi", "myös", "niin", "kun", "minä", "sinä", "ovat", "joka"]),
    ("pl", &["i", "w", "nie", "się", "na", "to", "jest", "że", "z", "do", "jak", "co", "ale", "tak", "o", "jestem", "czy", "dla", "być", "mnie"]),
    ("tr", &["ve", "bir", "bu", "da", "de", "ne", "için", "ile", "çok", "ben", "sen", "var", "yok", "değil", "gibi", "mi", "ama", "daha", "olarak", "nasıl"]),
    ("ms", &["dan", "yang", "di", "ini", "itu", "dengan", "untuk", "tidak", "saya", "anda", "ada", "akan", "dari", "dalam", "kami", "boleh", "juga", "apa", "ke", "pada"]),
    ("sw", &["na", "ya", "wa", "kwa", "ni", "katika", "hii", "kuwa", "za", "la", "cha", "huo", "lakini", "sana", "mimi", "wewe", "nini", "hapana", "ndiyo", "habari"]),
];

/// Letters that point to one language (each one found adds to its score).
const LETTERS: &[(char, &str)] = &[
    ('ñ', "es"), ('¿', "es"), ('¡', "es"), ('ç', "fr"), ('œ', "fr"), ('ß', "de"), ('ã', "pt"), ('õ', "pt"),
    ('å', "sv"), ('æ', "da"), ('ø', "no"), ('ą', "pl"), ('ę', "pl"), ('ł', "pl"), ('ś', "pl"), ('ź', "pl"), ('ż', "pl"),
    ('ğ', "tr"), ('ş', "tr"), ('ı', "tr"),
];

/// The language of `text`, as an ISO 639-1 code ("en" when unclear).
pub fn detect(text: &str) -> &'static str {
    // by script: the share of letters in each one
    let (mut letters, mut by) = (0usize, std::collections::HashMap::<&'static str, usize>::new());
    for c in text.chars().filter(|c| c.is_alphabetic()) {
        letters += 1;
        let s = match c as u32 {
            0x0900..=0x097F => "hi",   // Devanagari (Marathi and Nepali too: read as Hindi)
            0x0980..=0x09FF => "bn",
            0x0A00..=0x0A7F => "pa",
            0x0A80..=0x0AFF => "gu",
            0x0B00..=0x0B7F => "or",
            0x0B80..=0x0BFF => "ta",
            0x0C00..=0x0C7F => "te",
            0x0C80..=0x0CFF => "kn",
            0x0D00..=0x0D7F => "ml",
            0x0600..=0x06FF | 0x0750..=0x077F => "ar",
            0x0370..=0x03FF | 0x1F00..=0x1FFF => "el",
            0x0400..=0x04FF => "ru",
            0xAC00..=0xD7AF | 0x1100..=0x11FF | 0x3130..=0x318F => "ko",
            0x3040..=0x30FF => "ja",     // kana: Japanese, even with Han characters beside it
            0x4E00..=0x9FFF => "zh",
            0x0590..=0x05FF => "he",
            0x0E00..=0x0E7F => "th",
            _ => continue,
        };
        *by.entry(s).or_default() += 1;
    }
    if by.contains_key("ja") {
        return "ja";
    }
    if let Some((s, n)) = by.iter().max_by_key(|(_, n)| **n) {
        if *n * 3 >= letters.max(1) {
            return s;
        }
    }
    // Latin letters: common words and telling letters
    let lower = text.to_lowercase();
    let mut score: std::collections::HashMap<&'static str, f64> = std::collections::HashMap::new();
    for w in lower.split(|c: char| !(c.is_alphabetic() || c == '\'')).filter(|w| !w.is_empty()) {
        for (lang, words) in WORDS {
            if words.contains(&w) {
                *score.entry(lang).or_default() += 1.0;
            }
        }
    }
    for c in lower.chars() {
        if let Some((_, lang)) = LETTERS.iter().find(|(l, _)| *l == c) {
            *score.entry(lang).or_default() += 1.5;
        }
    }
    // English wins ties and unclear text
    let en = score.get("en").copied().unwrap_or(0.0);
    match score.into_iter().filter(|(l, _)| *l != "en").max_by(|a, b| a.1.total_cmp(&b.1)) {
        // two telling words, or one when nothing in the text is English ("Hola, ¿qué tal?")
        Some((lang, s)) if s > en && (s >= 2.0 || (en == 0.0 && s >= 1.0)) => lang,
        _ => "en",
    }
}

#[cfg(test)]
mod tests {
    use super::detect;

    #[test]
    fn finds_the_language_of_a_sentence() {
        let cases = [
            ("The model is running on your own computer, and nothing leaves it.", "en"),
            ("El modelo está funcionando en tu propio ordenador y no sale nada de él.", "es"),
            ("Le modèle tourne sur votre ordinateur, et rien ne sort de la machine.", "fr"),
            ("Das Modell läuft auf deinem eigenen Rechner, und nichts verlässt ihn.", "de"),
            ("Il modello è in esecuzione sul tuo computer e non esce niente.", "it"),
            ("O modelo está rodando no seu computador e não sai nada dele.", "pt"),
            ("Het model draait op je eigen computer en er gaat niets weg.", "nl"),
            ("Modellen körs på din egen dator och inget lämnar den.", "sv"),
            ("Model działa na twoim komputerze i nic nie wychodzi na zewnątrz.", "pl"),
            ("Model kendi bilgisayarınızda çalışıyor ve hiçbir şey dışarı çıkmıyor.", "tr"),
            ("मॉडल आपके अपने कंप्यूटर पर चल रहा है।", "hi"),
            ("النموذج يعمل على حاسوبك الخاص.", "ar"),
            ("Модель работает на вашем компьютере.", "ru"),
            ("Το μοντέλο τρέχει στον υπολογιστή σας.", "el"),
            ("모델이 당신의 컴퓨터에서 실행됩니다.", "ko"),
            ("মডেলটি আপনার নিজের কম্পিউটারে চলে।", "bn"),
            ("மாதிரி உங்கள் சொந்த கணினியில் இயங்குகிறது.", "ta"),
            ("మోడల్ మీ సొంత కంప్యూటర్‌లో నడుస్తుంది.", "te"),
            ("模型在你自己的电脑上运行。", "zh"),
            ("モデルはあなたのコンピューターで動いています。", "ja"),
            ("Hola. Este modelo funciona en tu propio ordenador.", "es"),
            ("Bonjour, merci!", "fr"),
            ("Ciao, grazie mille.", "it"),
            ("Hi!", "en"),
            ("OK", "en"),
            ("", "en"),
        ];
        for (text, want) in cases {
            assert_eq!(detect(text), want, "{text}");
        }
    }

    #[test]
    fn code_and_names_inside_english_stay_english() {
        assert_eq!(detect("Run `am2strata agentrun opencode` and the agent starts on your model."), "en");
        assert_eq!(detect("The answer is 42, see la_table in de_config for the details."), "en");
    }
}
