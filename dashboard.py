import gradio as gr
import json
import os
import pandas as pd
import plotly.express as px

def load_results():
    if not os.path.exists("evaluation_results.json"):
        return None

    with open(
        "evaluation_results.json",
        "r",
        encoding="utf-8"
    ) as f:
        content = f.read().strip()

    if not content:
        return None

    return json.loads(content)


def build_category_chart(data):
    rows = []

    for category, stats in data["category_stats"].items():
        accuracy = (
            stats["correct"] /
            stats["total"]
        ) * 100

        rows.append(
            {
                "Category": category,
                "Accuracy": accuracy
            }
        )

    df = pd.DataFrame(rows)

    fig = px.bar(
        df,
        x="Category",
        y="Accuracy",
        color="Accuracy",
        color_continuous_scale=[
            (0.0, "red"),
            (0.5, "yellow"),
            (1.0, "green")
        ]
    )

    return fig


def load_dashboard():
    data = load_results()
    chunk_text = ""
    if data is None:
        return (
            "No Results",
            "No Results",
            "No Results",
            "No Results",
            None,
            []
        )
    
    if len(data["detailed_results"]) > 0:
        first_result = data["detailed_results"][0]
        chunk_text = "\n\n".join(
            first_result["retrieved_chunks"]
        )


    failed_rows = []

    for item in data["failed_questions"]:
        failed_rows.append(
            [
                item["question"],
                item["category"],
                item["expected"],
                item["actual"]
            ]
        )

    return (
        f"{data['accuracy']:.2f}%",
        f"{data['coverage']:.2f}%",
        f"{data['recall']:.2f}%",
        f"{data['mrr']:.4f}",
        build_category_chart(data),
        build_judge_chart(data),
        failed_rows,
        chunk_text
    )

def build_judge_chart(data):

    df = pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Completeness",
                "Relevance"
            ],
            "Score": [
                data["judge_accuracy"],
                data["judge_completeness"],
                data["judge_relevance"]
            ]
        }
    )

    fig = px.bar(
        df,
        x="Metric",
        y="Score",
        color="Score",
        color_continuous_scale=[
            (0.0, "red"),
            (0.5, "yellow"),
            (1.0, "green")
        ]
    )

    return fig

with gr.Blocks() as demo:
    gr.Markdown("# Knowledge Worker Evaluation Dashboard")

    with gr.Row():
        accuracy = gr.Textbox(
            label="Accuracy"
        )

        coverage = gr.Textbox(
            label="Keyword Coverage"
        )

        recall = gr.Textbox(
            label="Recall@K"
        )

        mrr = gr.Textbox(
            label="MRR"
        )

    category_plot = gr.Plot(
        label="Category Performance"
    )
    judge_plot = gr.Plot(
        label="Judge Performance"
    )
    retrieved_chunks = gr.Textbox(
        label="Retrieved Chunks",
        lines=20
    )
    failed_df = gr.Dataframe(
        headers=[
            "Question",
            "Category",
            "Expected",
            "Actual"
        ]
    )
    
    question_details = gr.Dataframe(
        headers=[
            "Question",
            "Category",
            "Judge Accuracy",
            "Judge Completeness",
            "Judge Relevance"
        ]
    )
    load_btn = gr.Button(
        "Load Evaluation"
    )

    load_btn.click(
        fn=load_dashboard,
        outputs=[
            accuracy,
            coverage,
            recall,
            mrr,
            category_plot,
            judge_plot,
            failed_df,
            retrieved_chunks
        ]
    )

demo.launch()
