//! JSON response formats (`response_format` json_object / json_schema, the Responses API's `text.format`): the
//! model is asked once, in the system message, and the answer is checked before a client sees it. The engines have
//! no grammar decoder, so an answer that fails is an error, never a quiet retry. Port of serve/structured.py (with
//! the full schema check PolyStrata does when its optional jsonschema package is installed).

use serde_json::{json, Value};

pub struct Validator {
    schema: Option<jsonschema::Validator>,
}

/// An answer that is not the JSON asked for: 502 structured_output_failed.
pub struct StructuredError(pub String);

/// True when every value `node` accepts is a JSON object (type object, anyOf/oneOf of such, an allOf with one, or a
/// local $ref to one), so "return one JSON object" stays true.
fn only_objects(node: &Value, root: &Value, refs: &[String]) -> bool {
    let Some(o) = node.as_object() else { return false };
    if o.get("type").and_then(Value::as_str) == Some("object") {
        return true;
    }
    if let Some(r) = o.get("$ref").and_then(Value::as_str).filter(|r| r.starts_with('#') && !refs.contains(&r.to_string())) {
        let mut target = root;
        for part in r[1..].split('/').skip(1) {
            let part = part.replace("~1", "/").replace("~0", "~");
            match target.get(&part) {
                Some(t) => target = t,
                None => return false,
            }
        }
        let mut seen = refs.to_vec();
        seen.push(r.to_string());
        return only_objects(target, root, &seen);
    }
    for key in ["anyOf", "oneOf"] {
        if let Some(b) = o.get(key).and_then(Value::as_array).filter(|b| !b.is_empty()) {
            if b.iter().all(|x| only_objects(x, root, refs)) {
                return true;
            }
        }
    }
    o.get("allOf").and_then(Value::as_array).is_some_and(|b| b.iter().any(|x| only_objects(x, root, refs)))
}

fn check_refs(node: &Value) -> Result<(), String> {
    match node {
        Value::Object(o) => {
            for (k, v) in o {
                if (k == "$ref" || k == "$dynamicRef") && v.as_str().is_some_and(|s| !s.starts_with('#')) {
                    return Err("response_format supports only local schema references (#...)".into());
                }
                check_refs(v)?;
            }
            Ok(())
        }
        Value::Array(a) => a.iter().try_for_each(check_refs),
        _ => Ok(()),
    }
}

/// `response_format` -> the validator, with the format's directive added to the system message (`messages`: the
/// template's). None for plain text. Err: a 400. `with_tools`: the schema is for the final answer (Responses API).
pub fn prepare_format(rf: Option<&Value>, messages: &mut Vec<Value>, with_tools: bool) -> Result<Option<Validator>, String> {
    let Some(rf) = rf.filter(|v| !v.is_null()) else { return Ok(None) };
    let o = rf.as_object().ok_or("response_format must be an object")?;
    let (schema, full) = match o.get("type").and_then(Value::as_str) {
        Some("text") => return Ok(None),
        Some("json_object") => (json!({"type": "object"}), false),
        Some("json_schema") => {
            let spec = o.get("json_schema").and_then(Value::as_object).filter(|s| s.get("schema").is_some_and(Value::is_object))
                .ok_or("response_format.json_schema needs a schema object")?;
            if !spec.get("name").and_then(Value::as_str).is_some_and(|n| !n.is_empty()) {
                return Err("response_format.json_schema needs a name".into());
            }
            if spec.get("strict").is_some_and(|s| !s.is_boolean()) {
                return Err("response_format.json_schema.strict must be boolean".into());
            }
            let schema = spec["schema"].clone();
            if !only_objects(&schema, &schema, &[]) {
                return Err("response_format schema must accept only JSON objects at its root \
                            (type object, or anyOf/oneOf of object schemas)".into());
            }
            (schema, true)
        }
        _ => return Err("response_format.type must be text, json_object or json_schema".into()),
    };
    check_refs(&schema)?;
    let validator = if full {
        Some(jsonschema::validator_for(&schema).map_err(|e| format!("invalid response_format schema: {e}"))?)
    } else {
        None
    };
    let mut directive = format!(
        "OUTPUT FORMAT REQUIREMENT: Return exactly one JSON object matching the JSON Schema below. \
         No Markdown, headings, code fences, commentary, or text outside the JSON. \
         Use every required field, correct types, and only allowed fields. \
         Put all requested writing inside the appropriate JSON string fields.\nJSON Schema:\n{}",
        serde_json::to_string(&schema).unwrap_or_default()
    );
    if with_tools {
        directive.push_str("\nThis applies only to your final answer. To use a tool, call it as usual; the JSON object is \
                            what you write once you are done with the tools.");
    }
    match messages.first_mut().filter(|m| m["role"] == "system") {
        Some(m) => match &mut m["content"] {
            Value::Array(parts) => parts.push(json!({"type": "text", "text": directive})),
            c => *c = json!(format!("{}\n\n{directive}", c.as_str().unwrap_or(""))),
        },
        None => messages.insert(0, json!({"role": "system", "content": directive})),
    }
    Ok(Some(Validator { schema: validator }))
}

/// The JSON object in the answer: inside a ``` fence when there is one, from the first `{` to its match.
fn extract_json(text: &str) -> &str {
    let mut s = text.trim();
    if s.starts_with("```") {
        if let Some(first) = s.find('\n') {
            if let Some(end) = s[first + 1..].find("```") {
                s = s[first + 1..first + 1 + end].trim();
            }
        }
    }
    let Some(start) = s.find('{') else { return s };
    let (mut depth, mut in_str, mut esc) = (0i32, false, false);
    for (i, ch) in s[start..].char_indices() {
        if in_str {
            if esc {
                esc = false;
            } else if ch == '\\' {
                esc = true;
            } else if ch == '"' {
                in_str = false;
            }
        } else if ch == '"' {
            in_str = true;
        } else if ch == '{' {
            depth += 1;
        } else if ch == '}' {
            depth -= 1;
            if depth == 0 {
                return &s[start..start + i + 1];
            }
        }
    }
    &s[start..]
}

/// A JSON value that refuses duplicate keys (Python's object_pairs_hook check).
struct NoDup(Value);

impl<'de> serde::Deserialize<'de> for NoDup {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> serde::de::Visitor<'de> for V {
            type Value = NoDup;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("JSON")
            }
            fn visit_bool<E>(self, v: bool) -> Result<NoDup, E> { Ok(NoDup(json!(v))) }
            fn visit_i64<E>(self, v: i64) -> Result<NoDup, E> { Ok(NoDup(json!(v))) }
            fn visit_u64<E>(self, v: u64) -> Result<NoDup, E> { Ok(NoDup(json!(v))) }
            fn visit_f64<E>(self, v: f64) -> Result<NoDup, E> { Ok(NoDup(json!(v))) }
            fn visit_str<E>(self, v: &str) -> Result<NoDup, E> { Ok(NoDup(json!(v))) }
            fn visit_string<E>(self, v: String) -> Result<NoDup, E> { Ok(NoDup(json!(v))) }
            fn visit_unit<E>(self) -> Result<NoDup, E> { Ok(NoDup(Value::Null)) }
            fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut a: A) -> Result<NoDup, A::Error> {
                let mut out = Vec::new();
                while let Some(NoDup(v)) = a.next_element()? {
                    out.push(v);
                }
                Ok(NoDup(Value::Array(out)))
            }
            fn visit_map<A: serde::de::MapAccess<'de>>(self, mut a: A) -> Result<NoDup, A::Error> {
                let mut out = serde_json::Map::new();
                while let Some((k, NoDup(v))) = a.next_entry::<String, NoDup>()? {
                    if out.contains_key(&k) {
                        return Err(serde::de::Error::custom(format!("duplicate JSON key: {k}")));
                    }
                    out.insert(k, v);
                }
                Ok(NoDup(Value::Object(out)))
            }
        }
        d.deserialize_any(V)
    }
}

/// The checked answer as compact JSON, or why it is not what was asked for.
pub fn validated_json(text: &str, v: &Validator, finish: &str) -> Result<String, StructuredError> {
    if finish != "stop" {
        return Err(StructuredError(format!(
            "structured output was incomplete (finish_reason={finish}); increase the output budget")));
    }
    let NoDup(value) = serde_json::from_str::<NoDup>(extract_json(text))
        .map_err(|e| StructuredError(format!("model did not return valid JSON: {e}")))?;
    if !value.is_object() {
        return Err(StructuredError("model output failed the JSON Schema at /: the answer is not a JSON object".into()));
    }
    if let Some(s) = &v.schema {
        if let Some(e) = s.iter_errors(&value).next() {
            return Err(StructuredError(format!("model output failed the JSON Schema at {}: {e}",
                                               if e.instance_path.as_str().is_empty() { "/".to_string() } else { e.instance_path.to_string() })));
        }
    }
    Ok(serde_json::to_string(&value).unwrap_or_default())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schema() -> Value {
        json!({"type": "json_schema", "json_schema": {"name": "p", "schema": {"type": "object",
               "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}, "required": ["name", "age"]}}})
    }

    #[test]
    fn directive_and_checks() {
        let mut m = vec![json!({"role": "user", "content": "x"})];
        let v = prepare_format(Some(&schema()), &mut m, false).unwrap().unwrap();
        assert_eq!(m[0]["role"], "system");
        assert!(m[0]["content"].as_str().unwrap().contains("OUTPUT FORMAT REQUIREMENT"));
        assert_eq!(validated_json("```json\n{\"name\": \"A\", \"age\": 3}\n```", &v, "stop").ok().unwrap(), r#"{"name":"A","age":3}"#);
        assert!(validated_json(r#"{"name": "A"}"#, &v, "stop").is_err(), "missing required");
        assert!(validated_json(r#"{"name": "A", "age": "3"}"#, &v, "stop").is_err(), "wrong type");
        assert!(validated_json(r#"{"name": "A", "name": "B", "age": 1}"#, &v, "stop").is_err(), "duplicate key");
        assert!(validated_json(r#"{"name": "A", "age": 1}"#, &v, "length").is_err(), "cut off");
        let mut m2 = vec![];
        assert!(prepare_format(Some(&json!({"type": "json_schema", "json_schema": {"name": "a", "schema": {"type": "array"}}})), &mut m2, false).is_err());
        assert!(prepare_format(Some(&json!({"type": "text"})), &mut m2, false).unwrap().is_none());
    }
}
