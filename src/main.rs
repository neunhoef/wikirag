use async_openai::types::{
    ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
    CreateChatCompletionRequestArgs, CreateCompletionRequestArgs, CreateCompletionResponse,
};
use async_openai::Client;
use reqwest::Client as ReqClient;
use serde::Deserialize;
use std::io::{self, Write};

async fn get_keyword_from_chatgpt(question: &str) -> Result<String, Box<dyn std::error::Error>> {
    let client = Client::new();

    let request = CreateCompletionRequestArgs::default()
        .model("gpt-3.5-turbo-instruct")
        .prompt(format!("Extract exactly one keyword from this question for a Wikipedia lookup, respond with just the single keyword: {}", question))
        .max_tokens(32_u32)
        .build()?;

    let response: CreateCompletionResponse = client.completions().create(request).await?;

    println!("Token usage: {:?}", response.usage);

    if let Some(choice) = response.choices.first() {
        Ok(choice.text.trim().to_string())
    } else {
        Ok("No keywords found".to_string())
    }
}

async fn answer_question_with_wikipage(
    wikipage: &str,
    question: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let client = Client::new();

    let request = CreateChatCompletionRequestArgs::default()
        .max_tokens(200_u32)
        .model("gpt-4o")
        .messages([
            ChatCompletionRequestSystemMessageArgs::default()
                .content(format!("Take this text: {}", wikipage))
                .name("Wikipedia".to_string()).build()?.into(),
            ChatCompletionRequestUserMessageArgs::default()
                .content(format!(
                        "Now answer the following question, using the information in the provided text: {}",
                        question
                    ))
                .build()?.into(),
        ])
        .build()?;

    let response = client.chat().create(request).await?;
    println!("Token usage: {:?}", response.usage);

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

async fn search_wikipedia(
    keyword: &str,
) -> Result<(Vec<String>, Vec<String>), Box<dyn std::error::Error>> {
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

    //println!("Raw response: {}", body);

    let response: WikipediaResponse = serde_json::from_str(&body)?;

    let page_ids: Vec<String> = response
        .query
        .search
        .iter()
        .map(|result| result.pageid.to_string())
        .collect();

    let titles: Vec<String> = response
        .query
        .search
        .into_iter()
        .map(|result| result.title.to_string())
        .collect();
    Ok((page_ids, titles))
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

async fn download_wikipedia_page(page_id: &str) -> Result<String, Box<dyn std::error::Error>> {
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

    //println!("Raw response: {}", body);

    let response: WikipediaExtractResponse = serde_json::from_str(&body)?;

    if let Some(page) = response.query.pages.get(page_id) {
        Ok(page.extract.clone())
    } else {
        Err("Page not found".into())
    }
}

#[tokio::main]
async fn main() {
    print!("Enter your question: ");
    io::stdout().flush().unwrap();

    let mut question = String::new();
    io::stdin().read_line(&mut question).unwrap();

    let keyword: String = match get_keyword_from_chatgpt(&question.trim()).await {
        Ok(keyword) => {
            println!("Keywords: {}", keyword);
            keyword
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    };

    let res = search_wikipedia(&keyword).await;
    let (p, t) = match res {
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(2);
        }
        Ok(r) => r,
    };
    println!("Wikipedia search result: {:?} {:?}", p, t);

    // Download first page:
    let page = download_wikipedia_page(&p[0]).await;
    let p = match page {
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(3);
        }
        Ok(p) => p,
    };

    println!("Wikipedia page downloaded:\nSize: {}\n", p.len());
    let answer = answer_question_with_wikipage(&p, &question).await;
    match answer {
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(4);
        }
        Ok(a) => {
            println!("Answer: {}", a);
        }
    }
}
