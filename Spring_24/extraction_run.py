import fitz
import pandas as pd
import sys
from collections import Counter
import os
from glob import glob

class Extraction():
    """
    Takes in the source path (the path to the folder of PDFs) and the destination path (the path to the folder of TXTs) and:
    1. creates a df with information about each PDF file in the folder (index: file name, elements: source path, destination path, [FUTURE FEATURES: state, district, ...]);
    2. extracts the narrative for each pdf and saves each extraction as a TXT file, then saves the TXT file at the destination path.

    INPUTS: 
    source_path    the path to the folder of PDFs
    destination_path    the path to the folder of TXTs
    labels    a dataframe with the binary indicator (index: file_title, column: label)
    """
    source_path = ""
    destination_path = ""
    INFO = pd.DataFrame()

    def __init__(self, source_path, destination_path):
        self.source_path = source_path
        self.destination_path = destination_path

    def flags_decomposer(self, flags):
        """
        Make font flags human readable.
        """
        l = []
        if flags & 2 ** 0:
            l.append("superscript")
        if flags & 2 ** 1:
            l.append("italic")
        if flags & 2 ** 2:
            l.append("serifed")
        else:
            l.append("sans")
        if flags & 2 ** 3:
            l.append("monospaced")
        else:
            l.append("proportional")
        if flags & 2 ** 4:
            l.append("bold")
        return ", ".join(l)

    def get_narrative(self, pdf_path):
        """
        Given a pdf path, extracts the narrative of one pdf into a string.
        """
        doc = fitz.open(pdf_path)

        style_counts = []

        for page in doc:
            #, flags=11

            paths = page.get_drawings()  # get drawings on the page

            drawn_lines = []
            for p in paths:
                # print(p)
                for item in p["items"]:
                    # print(item[0])
                    if item[0] == "l":  # an actual line
                        # print(item[1], item[2])
                        p1, p2 = item[1], item[2]
                        if p1.y == p2.y:
                            drawn_lines.append((p1, p2))
                    elif item[0] == "re":  # a rectangle: check if height is small
                        # print(item[0])
                        # print(item[1])
                        r = item[1]
                        if r.width > r.height and r.height <= 2:
                            drawn_lines.append((r.tl, r.tr))  # take top left / right points

            blocks = page.get_text("dict", flags=11)["blocks"]

            for b in blocks:  # iterate through the text blocks
                for l in b["lines"]:  # iterate through the text lines
                    for s in l["spans"]:  # iterate through the text spans

                        font_properties = "Font: '%s' (%s), size %g, color #%06x" % (
                            s["font"],  # font name
                            self.flags_decomposer(s["flags"]),  # readable font flags
                            s["size"],  # font size
                            s["color"],  # font color
                        )

                        r = fitz.Rect(s['bbox'])
                        for p1, p2 in drawn_lines:  # check distances for start / end points
                            if abs(r.bl - p1) <= 4 and abs(r.br - p2) <= 4:
                                font_properties = " ".join([font_properties, 'underlined'])

                        style_counts.append(font_properties)

        styles = dict(Counter(style_counts))

        style_list = sorted(styles.items(), key=lambda x:x[1], reverse=True)

        headers = {}
        count = 0
        p_size = int(style_list[0][0].split('size')[1].split()[0].strip(','))

        for page in doc:
            #, flags=11
            blocks = page.get_text("dict", flags=11)["blocks"]

            for b in blocks:  # iterate through the text blocks
                for l in b["lines"]:  # iterate through the text lines
                    texts = ""
                    count+=1
                    for s in l['spans']:
                        if s['size'] >= p_size:
                            texts = "".join ([texts, s['text']])
                    text_list = texts.split()
                    if len(text_list) > 0 and len(text_list) < 7:
                        headers.update({texts:count})

        opinion_loc = headers['Opinion']

        count = 0
        p_size = int(style_list[0][0].split('size')[1].split()[0].strip(','))
        new_headers = {}
        header_properties = ""

        for page in doc:
            #, flags=11
            blocks = page.get_text("dict", flags=11)["blocks"]

            for b in blocks:  # iterate through the text blocks
                for l in b["lines"]:  # iterate through the text lines
                    count+=1
                    if count==opinion_loc:
                        for s in l['spans']:
                            header_properties = "Font: '%s' (%s), size %g, color #%06x" % (
                                s["font"],  # font name
                                self.flags_decomposer(s["flags"]),  # readable font flags
                                s["size"],  # font size
                                s["color"],  # font color
                            )

        count = 0
        for page in doc:
            #, flags=11
            blocks = page.get_text("dict", flags=11)["blocks"]

            for b in blocks:  # iterate through the text blocks
                for l in b["lines"]:  # iterate through the text lines
                    count+=1
                    for s in l['spans']:
                        font_properties = "Font: '%s' (%s), size %g, color #%06x" % (
                            s["font"],  # font name
                            self.flags_decomposer(s["flags"]),  # readable font flags
                            s["size"],  # font size
                            s["color"],  # font color
                        )
                        if font_properties==header_properties:
                            new_headers.update({s['text']:count})

        p_size = int(style_list[0][0].split('size')[1].split()[0].strip(','))
        p_color = style_list[0][0].split('color')[1].split()[0].strip(',')
        p_font = style_list[0][0]

        bad_fonts = []

        for style in style_list:
            font_str = style[0]
            s_size = int(font_str.split('size')[1].split()[0].strip(','))
            s_color = font_str.split('color')[1].split()[0].strip(',')

            # if font matches paragraph font, it's a bad_font
            if font_str==p_font:
                bad_fonts+=[font_str]
            # if font doesn't match paragraph text color, it's a bad_font
            if s_color!=p_color:
                bad_fonts+=[font_str]
            # if font matches characteristics of vocab word font, it's a bad font
            if ('bold' in font_str and 'underlined' in font_str) and ('italic' in font_str and p_size==s_size):
                bad_fonts+=[font_str]
            # if font size is smaller than paragraph text size, it's a bad_font
            if s_size<p_size:
                bad_fonts+=[font_str]

        master = []
        for style in style_list:
            if style[0] not in bad_fonts:
                master += [style[0]]

        for page in doc:

            paths = page.get_drawings()  # get drawings on the page

            drawn_lines = []
            for p in paths:
                # print(p)
                for item in p["items"]:
                    # print(item[0])
                    if item[0] == "l":  # an actual line
                        # print(item[1], item[2])
                        p1, p2 = item[1], item[2]
                        if p1.y == p2.y:
                            drawn_lines.append((p1, p2))
                    elif item[0] == "re":  # a rectangle: check if height is small
                        # print(item[0])
                        # print(item[1])
                        r = item[1]
                        if r.width > r.height and r.height <= 2:
                            drawn_lines.append((r.tl, r.tr))  # take top left / right points

        count = 0
        opinion_subheaders = {}
        p_color = style_list[0][0].split('color')[1].split()[0].strip(',')

        for page in doc:
            #, flags=11
            blocks = page.get_text("dict", flags=11)["blocks"]

            for b in blocks:  # iterate through the text blocks
                for l in b["lines"]:  # iterate through the text lines
                    texts = ""
                    count+=1
                    span_fonts = []
                    if count>=opinion_loc:
                        for s in l['spans']:
                            font_properties = "Font: '%s' (%s), size %g, color #%06x" % (
                                s["font"],  # font name
                                self.flags_decomposer(s["flags"]),  # readable font flags
                                s["size"],  # font size
                                s["color"],  # font color
                            )

                            r = fitz.Rect(s['bbox'])
                            for p1, p2 in drawn_lines:  # check distances for start / end points
                                if abs(r.bl - p1) <= 4 and abs(r.br - p2) <= 4:
                                    font_properties = " ".join([font_properties, 'underlined'])

                            span_fonts+=[font_properties]
                            texts = "".join ([texts, s['text']])

                    text_list = texts.split()
                    if len(text_list) > 0 and len(text_list) < 7:
                        if any(i in span_fonts for i in master):
                            opinion_subheaders.update({texts:count})
                        if texts.isupper()==True:
                            opinion_subheaders.update({texts:count})

        narrative = ""
        conclusion_loc = 100000
        count = 0
        p_size = int(style_list[0][0].split('size')[1].split()[0].strip(','))

        keys_as_list = list(opinion_subheaders)
        for header_index in range(len(keys_as_list)):
            header = keys_as_list[header_index]
            if 'conclusion' in header.lower():
                conclusion_loc = opinion_subheaders[header]

        for page in doc:
            #, flags=11
            blocks = page.get_text("dict", flags=11)["blocks"]

            for b in blocks:  # iterate through the text blocks
                for l in b["lines"]:  # iterate through the text lines
                    texts = ""
                    count+=1
                    if count>=opinion_loc and count < conclusion_loc:
                        for s in l['spans']:
                            if s['size'] == p_size:
                                texts = "".join ([texts, s['text']])

                    narrative = " ".join([narrative, texts])
        narrative = narrative.strip()
        
        return narrative

    def save_narrative_to_txt(self, narrative, pdf_path):
        """
        Saving the string 'narrative' as a txt file to the destination path.
        """
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        txt_path = os.path.join(self.destination_path, f"{base_name}.txt")

        with open(txt_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write(narrative)

    def process_pdfs(self):
        """
        Goes through all the pdf files in source path, extracts the narrative for each file, and saves each extraction as a txt file to the destination path. 
        """
        pdf_files = glob(os.path.join(self.source_path, "*.pdf"))
        narratives = []

        for pdf_path in pdf_files:
            narrative = self.get_narrative(pdf_path)
            narratives.append(narrative)
            # Save narrative to text file in the same folder, overwriting existing files
            self.save_narrative_to_txt(narrative, pdf_path)

    def create_info(self):
        """
        Creates a dataframe with information on each narrative file, assuming that the destination path already has txt files within the folder.
        """
        # outputs a list of all the txt files in the folder
        file_list = glob(os.path.join(self.destination_path, "*.txt"))
        file_data = []
        for each_file in file_list:
            # split might be different, recommend checking with INFO.sample() or .head()
            file_title = each_file.split('\\')[-1].split(".txt")[0]
            with open(each_file, 'r',  encoding='utf-8') as file:
                narrative = file.read()
            file_data.append((file_title, narrative))

        # creating df with the file title as the index and source path as a col
        INFO = pd.DataFrame(file_data, columns=['file_title', 'narrative'])\
            .set_index('file_title').sort_index()
        self.INFO = pd.concat([self.INFO, INFO])
        # attempt at dropping any duplicate files with same file name
        # this only works if same file has the SAME NAME
        self.INFO = self.INFO[~self.INFO.narrative.duplicated(keep='last')]
        return self.INFO

    def add_labels(self, labels):
        # add the labels
        labels.index = labels.index.str.replace("'", "_")
        self.INFO = pd.concat([self.INFO, labels], axis=1, join="inner")
        return self.INFO
    