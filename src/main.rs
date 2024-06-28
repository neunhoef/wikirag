use async_openai::types::{
    ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageArgs,
    ChatCompletionRequestUserMessageArgs, CompletionUsage, CreateChatCompletionRequestArgs,
};
use async_openai::Client;
use ollama_rs::{
    generation::chat::{request::ChatMessageRequest, ChatMessage},
    Ollama,
};
use reqwest::Client as ReqClient;
use serde::Deserialize;
use std::io;

enum LlmProvider {
    OpenAI,
    Ollama,
}

struct Config {
    pub model: String,
    pub verbose: bool,
    pub wiki_pages: u32,
    pub llm_server: LlmProvider,
}

fn get_config_from_env() -> Config {
    // Defaults:
    let mut c = Config {
        model: "gpt-3.5-turbo".into(),
        verbose: false,
        wiki_pages: 1,
        llm_server: LlmProvider::OpenAI,
    };
    if let Ok(val) = std::env::var("AI_MODEL") {
        match val.as_ref() {
            "gpt-4-turbo" | "gpt-3.5-turbo" | "gpt-4o" => {
                c.model = val;
                c.llm_server = LlmProvider::OpenAI;
            }
            "llama3" => {
                c.model = val;
                c.llm_server = LlmProvider::Ollama;
            }
            _ => {
                eprintln!(
                    "Unknown model {} requested, falling back to 'gpt-3.5-turbo'.
Only the following models are currently allowed:
  - gpt-4-turbo
  - gpt-4o
  - gpt-3.5-turbo
",
                    val
                );
            }
        }
    }
    if let Ok(val) = std::env::var("VERBOSE") {
        if !val.is_empty() {
            c.verbose = true;
        }
    }
    if let Ok(val) = std::env::var("WIKI_PAGES") {
        if !val.is_empty() {
            let n = val.parse::<u32>();
            if let Ok(n) = n {
                c.wiki_pages = n;
                if c.wiki_pages == 0 {
                    c.wiki_pages = 1;
                }
            }
        }
    }
    c
}

fn greet() {
    eprintln!(
        "This is WikiRag!

I will answer your question using knowledge from Wikipedia. I will first
use a LLM to derive key words to perform a search in Wikipedia and will
then retrieve the relevant pages. I will then feed these pages to the
LLM and let it answer your questions in this way. In the end you get the
answer plus a citation into Wikipedia.
"
    );
}

fn pretty_print_usage(config: &Config, usage: Option<CompletionUsage>) {
    if let Some(usage) = usage {
        let (in_costs, out_costs) = match config.model.as_ref() {
            "gpt-4-turbo" => (
                usage.prompt_tokens as f64 / 1_000_000.0 * 10.0,
                usage.completion_tokens as f64 / 1_000_000.0 * 30.0,
            ),
            "gpt-3.5-turbo" => (
                usage.prompt_tokens as f64 / 1_000_000.0 * 0.5,
                usage.completion_tokens as f64 / 1_000_000.0 * 1.5,
            ),
            "gpt-4o" => (
                usage.prompt_tokens as f64 / 1_000_000.0 * 5.0,
                usage.completion_tokens as f64 / 1_000_000.0 * 15.0,
            ),
            _ => (0.0, 0.0),
        };
        eprintln!(
            "Tokens in: {} (${:.6}), tokens out: {} (${:.6})",
            usage.prompt_tokens, in_costs, usage.completion_tokens, out_costs
        );
    }
}

async fn get_keywords_from_chatgpt(
    config: &Config,
    question: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let client = Client::new();

    let request = CreateChatCompletionRequestArgs::default()
        .max_tokens(32_u32)
        .model(&config.model)
        .messages([
            ChatCompletionRequestSystemMessageArgs::default()
                .content("Extract exactly one keyword from the user's question for a Wikipedia lookup, respond with just the single keyword.".to_string())
                .build()?.into(),
            ChatCompletionRequestUserMessageArgs::default()
                .content(question)
                .build()?.into(),
        ])
        .build()?;

    let response = client.chat().create(request).await?;

    pretty_print_usage(config, response.usage);

    if let Some(choice) = response.choices.first() {
        if let Some(msg) = &choice.message.content {
            Ok(msg.clone())
        } else {
            Ok("Did not receive response!".to_string())
        }
    } else {
        Ok("No keywords found".to_string())
    }
}

async fn answer_question_with_wikipage_openai(
    config: &Config,
    wikipage: &Vec<String>,
    question: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let client = Client::new();

    let mut messages: Vec<ChatCompletionRequestMessage> = vec![];
    for w in wikipage.iter() {
        messages.push(
            ChatCompletionRequestSystemMessageArgs::default()
                .content(w)
                .name("Wikipedia".to_string())
                .build()?
                .into(),
        );
    }
    messages.push(
        ChatCompletionRequestUserMessageArgs::default()
            .content(format!(
                "Now answer the following question, using the information in the provided text: {}",
                question
            ))
            .build()?
            .into(),
    );
    let request = CreateChatCompletionRequestArgs::default()
        .max_tokens(1000_u32)
        .model(&config.model)
        .messages(messages)
        .build()?;

    let response = client.chat().create(request).await?;

    pretty_print_usage(config, response.usage);

    if let Some(choice) = response.choices.first() {
        if let Some(msg) = &choice.message.content {
            Ok(msg.clone())
        } else {
            Ok("No response received".to_string())
        }
    } else {
        Ok("No keywords found".to_string())
    }
}

async fn get_keywords_from_ollama(
    config: &Config,
    question: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut ollama = Ollama::new_default_with_history(30);

    let user_msg = ChatMessage::system("Extract exactly one keyword from the user's question for a Wikipedia lookup, respond with just the single keyword. ".to_string() + question);

    let response = ollama
        .send_chat_messages_with_history(
            ChatMessageRequest::new(config.model.clone(), vec![user_msg]),
            "default".to_string(),
        )
        .await?;

    if let Some(msg) = response.message {
        Ok(msg.content.clone())
    } else {
        Ok("Did not receive response!".to_string())
    }
}

async fn answer_question_with_wikipage_ollama(
    config: &Config,
    wikipage: &Vec<String>,
    question: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut ollama = Ollama::new_default_with_history(30);

    let mut messages: String = "".to_string();
    for w in wikipage.iter() {
        messages.push_str(w);
        messages.push_str("\n");
    }
    messages.push_str(&format!(
        "Now answer the following question, using the information in the provided text: {}",
        question
    ));
    let user_msg = ChatMessage::system(messages);

    let response = ollama
        .send_chat_messages_with_history(
            ChatMessageRequest::new(config.model.clone(), vec![user_msg]),
            "default".to_string(),
        )
        .await?;

    if let Some(msg) = response.message {
        Ok(msg.content.clone())
    } else {
        Ok("No response received".to_string())
    }
}

#[derive(Deserialize, Debug)]
struct SearchResult {
    title: String,
    pageid: u32,
}

#[derive(Deserialize, Debug)]
struct QueryResult {
    search: Vec<SearchResult>,
}

#[derive(Deserialize, Debug)]
struct WikipediaResponse {
    query: QueryResult,
}

#[derive(Debug)]
struct WikiPage {
    pub page_id: String,
    pub title: String,
}

async fn search_wikipedia(
    config: &Config,
    keyword: &str,
) -> Result<Vec<WikiPage>, Box<dyn std::error::Error>> {
    let client = ReqClient::new();
    let base_url = "https://en.wikipedia.org/w/api.php";

    let params = [
        ("action", "query"),
        ("list", "search"),
        ("srsearch", keyword),
        ("format", "json"),
    ];

    let response = client.get(base_url).query(&params).send().await?;
    let body = response.text().await?;

    if config.verbose {
        eprintln!("Raw response: {}", body);
    }

    let response: WikipediaResponse = serde_json::from_str(&body)?;

    let pages: Vec<WikiPage> = response
        .query
        .search
        .iter()
        .map(|result| WikiPage {
            page_id: result.pageid.to_string(),
            title: result.title.to_string(),
        })
        .collect();

    Ok(pages)
}

#[derive(Deserialize, Debug)]
struct Page {
    extract: String,
}

#[derive(Deserialize, Debug)]
struct QueryPages {
    pages: std::collections::HashMap<String, Page>,
}

#[derive(Deserialize, Debug)]
struct WikipediaExtractResponse {
    query: QueryPages,
}

async fn download_wikipedia_page(
    config: &Config,
    page_id: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let client = ReqClient::new();
    let base_url = "https://en.wikipedia.org/w/api.php";

    let params = [
        ("action", "query"),
        ("pageids", page_id),
        ("prop", "extracts"),
        ("explaintext", "true"),
        ("format", "json"),
    ];

    let response = client.get(base_url).query(&params).send().await?;
    let body = response.text().await?;

    if config.verbose {
        eprintln!("Raw response: {}", body);
    }

    let response: WikipediaExtractResponse = serde_json::from_str(&body)?;

    if let Some(page) = response.query.pages.get(page_id) {
        Ok(page.extract.clone())
    } else {
        Err("Page not found".into())
    }
}

fn deal_with_error<T>(r: Result<T, Box<dyn std::error::Error>>, ec: i32) -> T {
    match r {
        Err(e) => {
            println!("Error: {}", e);
            std::process::exit(ec);
        }
        Ok(t) => t,
    }
}

#[tokio::main]
async fn main() {
    greet();

    let config = get_config_from_env();

    // Read question:
    let mut question = String::new();
    eprintln!("Please enter your question:");
    io::stdin().read_line(&mut question).unwrap();

    eprintln!(
        "\nPerforming keyword derivation using LLM model {}...",
        config.model
    );
    let res = match config.llm_server {
        LlmProvider::OpenAI => get_keywords_from_chatgpt(&config, &question.trim()).await,
        LlmProvider::Ollama => get_keywords_from_ollama(&config, &question.trim()).await,
    };
    let keywords: String = deal_with_error(res, 1);
    eprintln!("Keywords found: {}", keywords);

    eprintln!("\nPerforming lookup in Wikipedia using '{}'...", keywords);
    let res = search_wikipedia(&config, &keywords).await;
    let pages = deal_with_error(res, 2);
    eprintln!("Wikipedia search results:");
    eprintln!("  page id | title");
    eprintln!("==========|===============================");
    for p in pages.iter() {
        eprintln!("{:>10}| {}", p.page_id, p.title);
    }
    eprintln!("");

    // Download pages:
    let mut page_strings: Vec<String> = vec![];
    for i in 0..config.wiki_pages as usize {
        if i >= pages.len() {
            break;
        }
        let res = download_wikipedia_page(&config, &pages[i].page_id).await;
        let page = deal_with_error(res, 3);
        eprintln!(
            "Wikipedia page downloaded '{}': Size: {}",
            pages[i].title,
            page.len(),
        );
        page_strings.push(page);
    }

    eprintln!("\nAnswering question using Wikipedia pages and LLM model...");
    let res = match config.llm_server {
        LlmProvider::OpenAI => {
            answer_question_with_wikipage_openai(&config, &page_strings, &question).await
        }
        LlmProvider::Ollama => {
            answer_question_with_wikipage_ollama(&config, &page_strings, &question).await
        }
    };
    let answer = deal_with_error(res, 4);
    eprintln!("\nAnswer:");
    println!("{}", answer);
}
