#!/usr/bin/env python3
"""Vocabulary Workbench — describe what you feel, engine learns from you.

No numbers to fill in. Just read the sentence and write what you feel.
The engine score is shown for reference so you can see what IT reads.

Usage:
    python3 tools/vocab_workbench.py
    python3 tools/vocab_workbench.py --port 8080
    python3 tools/vocab_workbench.py --generate
"""

import argparse, json, os, sys, time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
import urllib.parse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

DATA_DIR = Path(__file__).parent / "workbench_data"
SENTENCES_FILE = DATA_DIR / "sentences.json"
RESPONSES_FILE = DATA_DIR / "responses.json"


def generate_sentences():
    from engine.pendulum import compute_vadug

    raw = [
        ("shes pregnant", "life_event"),
        ("we are pregnant!", "life_event"),
        ("shes 15 and pregnant", "life_event"),
        ("my daughter is pregnant", "life_event"),
        ("he was diagnosed with cancer", "life_event"),
        ("he was diagnosed early and its treatable", "life_event"),
        ("she was adopted", "life_event"),
        ("we adopted a baby!", "life_event"),
        ("she overdosed last night", "life_event"),
        ("3 years sober today", "life_event"),
        ("he relapsed again", "life_event"),
        ("they got divorced", "life_event"),
        ("he got laid off", "life_event"),
        ("we were just sitting there when it happened", "contrast"),
        ("everything was fine and then the phone rang", "contrast"),
        ("she was just walking home from school", "contrast"),
        ("he was sleeping when they broke in", "contrast"),
        ("we were laughing five minutes ago", "contrast"),
        ("you stupid bitch", "directed"),
        ("shes a bad bitch", "directed"),
        ("you lucky bastard", "directed"),
        ("that was savage", "directed"),
        ("hes a legend", "directed"),
        ("youre a monster", "directed"),
        ("ok boomer", "directed"),
        ("she lost a lot of weight", "body"),
        ("hes getting fat", "body"),
        ("you look different", "body"),
        ("she doesnt eat anymore", "body"),
        ("you look like shit", "body"),
        ("bro that was fire", "slang"),
        ("she absolutely killed it", "slang"),
        ("im dead lol", "slang"),
        ("no cap that was insane", "slang"),
        ("hes cooked", "slang"),
        ("lowkey goated", "slang"),
        ("skill issue", "slang"),
        ("cope harder", "slang"),
        ("touch grass", "slang"),
        ("thats mid", "slang"),
        ("i miss my ex", "relationship"),
        ("we used to be best friends", "relationship"),
        ("she moved on already", "relationship"),
        ("he ghosted me", "relationship"),
        ("we broke up", "relationship"),
        ("i think im in love", "relationship"),
        ("its getting dark outside", "atmosphere"),
        ("the house was quiet", "atmosphere"),
        ("the room felt heavy", "atmosphere"),
        ("it smelled like home", "atmosphere"),
        ("everything was still", "atmosphere"),
        ("come on mom its only 15 dollars", "numbers"),
        ("shes 15 and pregnant", "numbers"),
        ("married for 40 years", "numbers"),
        ("we lost 47 people", "numbers"),
        ("clean for 6 months", "numbers"),
        ("im fine", "passive_agg"),
        ("whatever you say", "passive_agg"),
        ("no its fine i didnt want to go anyway", "passive_agg"),
        ("sure go ahead", "passive_agg"),
        ("i guess i deserved it", "passive_agg"),
        ("oh great another meeting", "sarcasm"),
        ("wow thanks so much", "sarcasm"),
        ("yeah that went well", "sarcasm"),
        ("love that for you", "sarcasm"),
        ("i dont want to be here anymore", "crisis"),
        ("nobody would even notice", "crisis"),
        ("im just tired", "crisis"),
        ("it doesnt matter", "crisis"),
        ("i cant do this anymore", "crisis"),
        ("just leave me alone", "crisis"),
        ("whats the point", "crisis"),
        ("i want to kill myself", "crisis"),
        ("tell my kids i love them", "crisis"),
        ("im going to jump", "crisis"),
        ("i have the pills right here", "crisis"),
        ("tonight is the night", "crisis"),
        ("i wrote a note", "crisis"),
        ("my wife cheated on me with my best friend", "betrayal"),
        ("he told everyone my secret", "betrayal"),
        ("she took the kids and left", "betrayal"),
        ("they went behind my back", "betrayal"),
        ("i trusted him with everything", "betrayal"),
        ("im worthless", "self_worth"),
        ("i am nothing", "self_worth"),
        ("im just a burden", "self_worth"),
        ("why would anyone love me", "self_worth"),
        ("im proud of what we built", "self_worth"),
        ("i finally feel like i belong", "self_worth"),
        ("i was the only one at my birthday", "only"),
        ("i was the only one to finish the test", "only"),
        ("its only 15 dollars", "only"),
        ("i only paid 15 dollars for this and its worth 60 online", "only"),
    ]

    sentences = []
    for idx, (text, category) in enumerate(raw):
        try:
            vadug, meta = compute_vadug(text)
            structs = [s.pattern for s in meta.get("structures", [])]
            engine = {
                "v": vadug.v, "a": vadug.a, "d": vadug.d,
                "u": vadug.u, "g": vadug.g, "w": vadug.w, "i": vadug.i,
                "structures": structs,
            }
        except Exception as e:
            engine = {"error": str(e)}
        sentences.append({"id": idx, "text": text, "category": category, "engine": engine})
    return sentences


def load_responses():
    if RESPONSES_FILE.exists():
        with open(RESPONSES_FILE) as f:
            return json.load(f)
    return {}


def save_responses(responses):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESPONSES_FILE, "w") as f:
        json.dump(responses, f, indent=2)


def build_html(sentences, responses):
    s_json = json.dumps(sentences).replace("</", "<\\/")
    r_json = json.dumps(responses).replace("</", "<\\/")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>VADUGWI — What Do You Feel?</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #0a0a0a; color: #e0e0e0;
    padding: 12px; max-width: 700px; margin: 0 auto;
}}
h1 {{ color: #00ff88; font-size: 1.2em; padding: 10px 0 3px; }}
.sub {{ color: #666; font-size: 0.8em; padding-bottom: 10px; border-bottom: 1px solid #222; }}
.stats {{ color: #888; font-size: 0.85em; padding: 8px 0; }}
.stats span {{ color: #00ff88; }}
.filters {{
    display: flex; gap: 6px; flex-wrap: wrap; padding: 8px 0 12px;
}}
.fbtn {{
    background: #1a1a1a; border: 1px solid #333; color: #aaa;
    padding: 5px 10px; border-radius: 4px; cursor: pointer;
    font-size: 0.75em; font-family: inherit;
}}
.fbtn:hover {{ border-color: #00ff88; color: #00ff88; }}
.fbtn.on {{ background: #00ff8822; border-color: #00ff88; color: #00ff88; }}
.bar {{
    position: sticky; top: 0; background: #0a0a0a; padding: 6px 0;
    z-index: 10; border-bottom: 1px solid #222;
    display: flex; justify-content: space-between;
}}
.card {{
    background: #111; border: 1px solid #222; border-radius: 8px;
    padding: 14px; margin-bottom: 10px;
}}
.card.done {{ display: none; }}
.card.skip {{ display: none; }}
.sentence {{ font-size: 1.15em; color: #fff; font-style: italic; margin-bottom: 6px; }}
.tag {{
    display: inline-block; background: #1a1a2a; color: #6688ff;
    padding: 1px 7px; border-radius: 3px; font-size: 0.7em; margin-bottom: 6px;
}}
.engine {{ color: #555; font-size: 0.75em; margin-bottom: 8px; font-family: monospace; }}
textarea {{
    width: 100%; background: #0a0a0a; border: 1px solid #333;
    color: #e0e0e0; padding: 10px; border-radius: 4px;
    font-family: inherit; font-size: 0.95em;
    resize: vertical; min-height: 60px;
}}
textarea:focus {{ border-color: #00ff88; outline: none; }}
textarea::placeholder {{ color: #444; }}
.btns {{ display: flex; gap: 8px; margin-top: 6px; align-items: center; }}
.sv {{
    background: #00ff8822; border: 1px solid #00ff88; color: #00ff88;
    padding: 5px 14px; border-radius: 4px; cursor: pointer;
    font-family: inherit; font-size: 0.85em;
}}
.sv:hover {{ background: #00ff8844; }}
.sk {{
    background: transparent; border: 1px solid #444; color: #666;
    padding: 5px 14px; border-radius: 4px; cursor: pointer;
    font-family: inherit; font-size: 0.85em;
}}
.ind {{ color: #00ff88; font-size: 0.8em; opacity: 0; transition: opacity 0.3s; }}
.ind.show {{ opacity: 1; }}
@media (max-width: 600px) {{
    body {{ padding: 6px; }}
    .card {{ padding: 10px; }}
    textarea {{ font-size: 1em; min-height: 80px; }}
}}
</style>
</head>
<body>
<h1>What Do You Feel?</h1>
<div class="sub">Read the sentence. Write what you feel. That's it.</div>
<div class="stats">
    <span id="dc">0</span> done / <span id="tc">0</span>
    &middot; <span id="sc">0</span> skipped
</div>
<div class="filters" id="fl"></div>
<div class="bar">
    <button class="fbtn" onclick="jump()">Next unmapped</button>
    <button class="fbtn" id="togbtn" onclick="togDone()">Show completed</button>
    <button class="fbtn" onclick="exp()">Export</button>
</div>
<div id="cards"></div>
<script>
"use strict";
var S={s_json}, R={r_json}, F='all', showDone=false;
function init(){{ bfil(); render(); stats(); }}
function bfil(){{
    var c=['all'],seen={{}};
    S.forEach(function(s){{ if(!seen[s.category]){{ seen[s.category]=1; c.push(s.category); }} }});
    var el=document.getElementById('fl');
    while(el.firstChild) el.removeChild(el.firstChild);
    c.forEach(function(cat){{
        var b=document.createElement('button');
        b.className='fbtn'+(cat===F?' on':'');
        b.textContent=cat;
        b.addEventListener('click',function(){{ F=cat; bfil(); render(); }});
        el.appendChild(b);
    }});
}}
function render(){{
    var el=document.getElementById('cards');
    while(el.firstChild) el.removeChild(el.firstChild);
    var list=F==='all'?S:S.filter(function(s){{ return s.category===F; }});
    list.forEach(function(s){{
        var r=R[s.id]||{{}};
        var d=document.createElement('div');
        d.className='card'+(r.status==='done'?' done':'')+(r.status==='skip'?' skip':'');
        d.id='c-'+s.id;
        var tg=document.createElement('span'); tg.className='tag'; tg.textContent=s.category;
        d.appendChild(tg);
        var sn=document.createElement('div'); sn.className='sentence';
        sn.textContent='"'+s.text+'"';
        d.appendChild(sn);
        var en=document.createElement('div'); en.className='engine';
        var e=s.engine;
        if(e.error){{ en.textContent='Engine: ERROR'; }}
        else{{ en.textContent='Engine reads: V='+e.v+' A='+e.a+' D='+e.d+' W='+e.w+' I='+e.i+(e.structures.length?' ['+e.structures.join(', ')+']':''); }}
        d.appendChild(en);
        var ta=document.createElement('textarea');
        ta.id='t-'+s.id;
        ta.placeholder='What do you feel when you read this? What words carry the weight? What would change the meaning?';
        ta.value=r.feeling||'';
        d.appendChild(ta);
        var row=document.createElement('div'); row.className='btns';
        var sv=document.createElement('button'); sv.className='sv'; sv.textContent='Save';
        sv.addEventListener('click',function(){{ save(s.id); }});
        row.appendChild(sv);
        var sk=document.createElement('button'); sk.className='sk'; sk.textContent='Skip';
        sk.addEventListener('click',function(){{ skip(s.id); }});
        row.appendChild(sk);
        var ind=document.createElement('span'); ind.className='ind'; ind.id='i-'+s.id; ind.textContent='saved';
        row.appendChild(ind);
        d.appendChild(row);
        el.appendChild(d);
    }});
}}
function save(id){{
    var txt=document.getElementById('t-'+id).value;
    R[id]={{feeling:txt,status:'done',ts:new Date().toISOString()}};
    var x=new XMLHttpRequest();
    x.open('POST','/api/save');
    x.setRequestHeader('Content-Type','application/json');
    x.onload=function(){{
        var ind=document.getElementById('i-'+id);
        ind.textContent='saved'; ind.classList.add('show');
        setTimeout(function(){{ ind.classList.remove('show'); }},1500);
        document.getElementById('c-'+id).className='card done';
        stats();
    }};
    x.send(JSON.stringify({{id:id,data:R[id]}}));
}}
function skip(id){{
    R[id]={{status:'skip',ts:new Date().toISOString()}};
    var x=new XMLHttpRequest();
    x.open('POST','/api/save');
    x.setRequestHeader('Content-Type','application/json');
    x.onload=function(){{
        document.getElementById('c-'+id).className='card skip';
        stats();
    }};
    x.send(JSON.stringify({{id:id,data:R[id]}}));
}}
function stats(){{
    var dn=0,sk=0;
    for(var k in R){{ if(R[k].status==='done')dn++; if(R[k].status==='skip')sk++; }}
    document.getElementById('dc').textContent=dn;
    document.getElementById('tc').textContent=S.length;
    document.getElementById('sc').textContent=sk;
}}
function jump(){{
    var list=F==='all'?S:S.filter(function(s){{ return s.category===F; }});
    for(var i=0;i<list.length;i++){{
        if(!R[list[i].id]||R[list[i].id].status!=='done'){{
            document.getElementById('c-'+list[i].id).scrollIntoView({{behavior:'smooth',block:'center'}});
            return;
        }}
    }}
}}
function togDone(){{
    showDone=!showDone;
    document.getElementById('togbtn').textContent=showDone?'Hide completed':'Show completed';
    var cards=document.querySelectorAll('.card.done,.card.skip');
    for(var i=0;i<cards.length;i++) cards[i].style.display=showDone?'block':'none';
}}
function exp(){{
    var x=new XMLHttpRequest(); x.open('GET','/api/export'); x.responseType='blob';
    x.onload=function(){{
        var a=document.createElement('a');
        a.href=URL.createObjectURL(x.response);
        a.download='feelings_'+new Date().toISOString().slice(0,10)+'.json';
        a.click();
    }};
    x.send();
}}
init();
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    sentences = []
    responses = {}

    def do_GET(self):
        p = urllib.parse.urlparse(self.path).path
        if p in ("/", ""):
            page = build_html(self.sentences, self.responses)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(page.encode())
        elif p == "/api/export":
            export = []
            for s in self.sentences:
                r = self.responses.get(str(s["id"]), {})
                if r.get("status") == "done":
                    export.append({
                        "text": s["text"],
                        "category": s["category"],
                        "engine": s["engine"],
                        "feeling": r.get("feeling", ""),
                        "timestamp": r.get("ts", ""),
                    })
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Disposition", "attachment")
            self.end_headers()
            self.wfile.write(json.dumps(export, indent=2).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode()
        if self.path == "/api/save":
            try:
                payload = json.loads(body)
                self.responses[str(payload["id"])] = payload["data"]
                save_responses(self.responses)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"ok":true}')
            except Exception as e:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *a):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.generate or not SENTENCES_FILE.exists():
        print("Generating sentences...")
        sentences = generate_sentences()
        with open(SENTENCES_FILE, "w") as f:
            json.dump(sentences, f, indent=2)
        print(f"{len(sentences)} sentences ready")
    else:
        with open(SENTENCES_FILE) as f:
            sentences = json.load(f)
        print(f"Loaded {len(sentences)} sentences")

    responses = load_responses()
    done = sum(1 for r in responses.values() if r.get("status") == "done")
    print(f"{done}/{len(sentences)} mapped")

    Handler.sentences = sentences
    Handler.responses = responses

    server = HTTPServer(("0.0.0.0", args.port), Handler)
    print(f"\nhttp://localhost:{args.port}")
    print("Ctrl+C to stop\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDone.")
        server.shutdown()


if __name__ == "__main__":
    main()
