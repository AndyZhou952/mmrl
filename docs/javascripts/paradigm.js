// Paradigm-family auto-detection for the Diffusion-RL Survey.
//
// The survey's taxonomy splits every algorithm into two families:
//   - "Policy Gradient"   (terracotta accent)
//   - "Direct Preference" (muted teal-green accent)
//
// The family is declared in prose only — the "Paradigm" row of each page's
// metadata table (e.g. `| **Paradigm** | **Policy Gradient** — ... |`). The
// markdown is GitHub-rendered too and cannot be touched, and the built DOM
// carries no per-page family marker. CSS alone cannot branch on cell *text*,
// so this script reads that cell at runtime and tags the page; all family
// theming then lives in CSS keyed off the resulting attribute.
//
// It sets `data-paradigm="policy-gradient" | "direct-preference"` on the
// page's <article class="md-content__inner">. Scoping to the article (not
// <body>) means the value naturally resets per page under Material's instant
// navigation — there is no stale attribute to clear. document$ re-fires on
// every navigation, so this stays correct as the user moves between pages.

(function () {
  function classify(text) {
    var t = (text || "").toLowerCase();
    if (t.indexOf("policy gradient") !== -1) return "policy-gradient";
    if (t.indexOf("direct preference") !== -1) return "direct-preference";
    return null;
  }

  function tagParadigm(root) {
    var article = root.querySelector(".md-content__inner");
    if (!article) return;

    // Find the metadata "spec-sheet" table: a 2-column "Field | Value" table
    // whose rows include a "Paradigm" field. Tag it so CSS can style it
    // distinctly from ordinary comparison/Limitations tables (which also appear
    // first on some reference pages — so a positional :first-of-type selector
    // alone would be wrong). The family value is read from the same scan.
    var family = null;
    var tables = article.querySelectorAll("table");
    for (var ti = 0; ti < tables.length; ti++) {
      var rows = tables[ti].querySelectorAll("tbody tr");
      var fam = null;
      for (var i = 0; i < rows.length; i++) {
        var cells = rows[i].cells;
        if (cells && cells.length >= 2) {
          var label = (cells[0].textContent || "").trim().toLowerCase();
          if (label === "paradigm") {
            fam = classify(cells[1].textContent);
          }
        }
      }
      if (fam) {
        tables[ti].setAttribute("data-meta-table", "");
        family = fam;
        break; // only the first matching table is the metadata header
      }
    }

    if (family) {
      article.setAttribute("data-paradigm", family);
    } else {
      article.removeAttribute("data-paradigm");
    }

    // Editorial run-in labels: a paragraph beginning with a bold label and a
    // colon (Issue / Idea / Why this works / Result). All are styled as
    // small-caps run-ins via CSS (p > strong:first-child). "Result" is the
    // empirical payoff and earns extra emphasis — but CSS cannot branch on
    // text, so tag those paragraphs here. (Purely additive; if this script
    // never runs, the run-ins still get the shared small-caps treatment.)
    var paras = article.querySelectorAll("p");
    for (var pi = 0; pi < paras.length; pi++) {
      var p = paras[pi];
      var strong = p.firstElementChild;
      if (
        strong &&
        strong.tagName === "STRONG" &&
        strong === p.firstChild &&
        /^result\b/i.test((strong.textContent || "").trim())
      ) {
        p.setAttribute("data-runin", "result");
      }
    }
  }

  if (typeof document$ !== "undefined" && document$.subscribe) {
    document$.subscribe(function (ctx) {
      tagParadigm((ctx && ctx.body) || document.body);
    });
  } else {
    document.addEventListener("DOMContentLoaded", function () {
      tagParadigm(document.body);
    });
  }
})();
