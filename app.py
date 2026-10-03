"""Local retrospective benchmark explorer. Run: python app.py"""
import gradio as gr
from src.demo import resources, predict


def compare(peptide, allele):
    try:
        return predict(peptide, allele)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc


def build_app():
    df,_=resources()
    counts=df.groupby('allele').size().sort_values()
    examples=[]
    for allele in [counts.index[-1],counts.index[0]]:
        row=df[df.allele==allele].iloc[0]
        examples.append([row.peptide,allele])
    with gr.Blocks(title='Peptide–HLA stability: what does pretraining add?') as demo:
        gr.Markdown('# What does protein pretraining add?\nCompare a sequence-trained neural network with frozen ESM-2 representations. '
                    'This is a retrospective research demo for 9-mer, class-I peptide–HLA pairs.')
        with gr.Row():
            peptide=gr.Textbox(label='Peptide (9 amino acids)',value=examples[0][0])
            allele=gr.Dropdown(choices=sorted(counts.index),value=examples[0][1],label='HLA allele')
        gr.Markdown('Dataset examples use cached representations. A new peptide loads ESM-2 locally on first use and may take several minutes.')
        submit=gr.Button('Compare predictions',variant='primary')
        card=gr.Markdown()
        table=gr.Dataframe(interactive=False,label='Predictions, calibrated baseline interval and fold sensitivity')
        gr.Examples(examples=examples,inputs=[peptide,allele],outputs=[card,table],fn=compare,cache_examples=False,
                    label='Try a well-measured or sparsely measured allele in this dataset',run_on_click=True)
        submit.click(compare,[peptide,allele],[card,table])
        demo.load(compare,[peptide,allele],[card,table])
        gr.Markdown('Zeros are retained as reported, although they may be censored. Engineered C67S constructs are excluded. '
                    'Only the separately calibrated baseline has a 95% target prediction interval. The five-fit ranges are model-sensitivity diagnostics. '
                    'Dataset counts are not population frequencies. No laboratory savings or clinical utility are established.')
    return demo


if __name__=='__main__':
    build_app().launch(server_name='127.0.0.1',server_port=7860,share=False)
