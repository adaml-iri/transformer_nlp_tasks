import pysbd
from itertools import islice
import argparse


class NLP_Preprocessing_Segmentation:
    def sentence_splitter(lang, infile, output_dir):
        seg = pysbd.Segmenter(language=lang, clean=False)
        N = 100
        count = 0
        text = ""
        out_file = open(output_dir, "a")
        in_file = open(infile, 'r')
        line = in_file.readline()
        while line != "":
            text = text + line
            count +=1
            if (count == N) :
                count = 0
                list_of_sentences = seg.segment(text)
                for sentence in list_of_sentences:
                    sentence = sentence.strip()
                    out_file.write(sentence+"\n")
                text = ""
            line = in_file.readline()
        else :
            in_file.close()
            out_file.close()



if __name__ == "__main__":
    nlp = NLP_Preprocessing_Segmentation()
    parser = argparse.ArgumentParser(description="Utility to detect and split text by senetence.")
    parser.add_argument("language", type=str, help="Provide language model should detect with.")
    parser.add_argument("infile", type=str, help="Provide full path to text file.")
    parser.add_argument("output_dir", type=str, help="Provide full path to directory where output will be placed.")
    args = parser.parse_args()
    ready = True
    lang = args.language
    infile = args.infile
    output_dir = args.output_dir
    if lang == "" or lang == None:
        lang = "en"
    if output_dir == "" or output_dir == None:
        print("Please provide directory to output.")
        ready = False
    if infile == "" or infile == None:
        print("Please provide path to text file.")
        ready = False
    if ready:
        nlp.sentence_splitter(lang, infile, output_dir)