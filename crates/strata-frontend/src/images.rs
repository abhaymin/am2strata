//! Pictures in a prompt: their sources in prompt order, and each one's placeholder expanded to its token count.
//! Port of serve/frontend.py `images_of` and the expansion in serve/server.py `Service.prepare`.

use serde_json::Value;

/// The image sources of the rendered conversation, in prompt order (one image item per picture).
pub fn images_of(messages: &[Value]) -> Vec<String> {
    messages
        .iter()
        .filter_map(|m| m.get("content").and_then(Value::as_array))
        .flatten()
        .filter(|item| item.get("type").and_then(Value::as_str) == Some("image"))
        .map(|item| item.get("source").and_then(Value::as_str).unwrap_or("").to_string())
        .collect()
}

/// One `pad` per picture -> one per image token. Only a pad right after `start` (the marker the template writes
/// for an image) is a picture; a pad elsewhere (the text of a message that quotes the marker) becomes the ids of
/// its text, `literal`. `counts`: each picture's token count, in prompt order.
pub fn expand_images(ids: &[u32], pad: u32, start: Option<u32>, literal: &[u32], counts: &[usize])
    -> Result<Vec<u32>, String> {
    let mut out = Vec::with_capacity(ids.len() + counts.iter().sum::<usize>());
    let mut k = 0;
    for (j, &t) in ids.iter().enumerate() {
        let is_image = t == pad
            && match start {
                None => k < counts.len(),
                Some(s) => j > 0 && ids[j - 1] == s,
            };
        if is_image {
            if k == counts.len() {
                return Err("the prompt and its images do not match".into());
            }
            out.extend(std::iter::repeat_n(pad, counts[k]));
            k += 1;
        } else if t == pad {
            out.extend_from_slice(literal);
        } else {
            out.push(t);
        }
    }
    if k != counts.len() {
        return Err("the prompt and its images do not match".into());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expands_only_marked_pads() {
        // 5 = vision_start, 9 = image_pad; the second 9 is quoted text (not after a start)
        let ids = [1, 5, 9, 6, 2, 9, 3];
        let out = expand_images(&ids, 9, Some(5), &[70, 71], &[3]).unwrap();
        assert_eq!(out, vec![1, 5, 9, 9, 9, 6, 2, 70, 71, 3]);
        assert!(expand_images(&ids, 9, Some(5), &[70], &[3, 4]).is_err());
    }

    #[test]
    fn finds_sources_in_order() {
        let m = serde_json::json!([
            {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "image", "source": "data:1"}]},
            {"role": "tool", "content": [{"type": "image", "source": "data:2"}]}]);
        assert_eq!(images_of(m.as_array().unwrap()), vec!["data:1", "data:2"]);
    }
}
