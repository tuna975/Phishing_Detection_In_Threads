"""
Pearson Correlation Analysis — Thread-Level Phishing Features
==============================================================
Computes pairwise Pearson correlation between all 12 features
and generates heatmaps for:
  1. Full dataset (all splits combined)
  2. Legitimate threads only
  3. Attack threads only
  4. Feature vs Label correlation (which features discriminate best)
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

# Global font settings — tuned for two-column LaTeX (figure spans full textwidth)
plt.rcParams.update({
    'font.size':          12,
    'font.weight':        'bold',
    'axes.titleweight':   'bold',
    'axes.labelweight':   'bold',
    'axes.titlesize':     16,
    'axes.labelsize':     13,
    'xtick.labelsize':    12,
    'ytick.labelsize':    12,
    'legend.fontsize':    11,
    'legend.title_fontsize': 12,
    'figure.dpi':         300,
    'savefig.dpi':        300,
    'pdf.fonttype':       42,   # embed fonts for PDF
    'ps.fonttype':        42,
})

# ============================================================================
# CONFIGURATION
# ============================================================================

FEATURE_COLS = [
    'response_time_variance', 'thread_length_days', 'timestamp_anomalies',
    'tone_shift', 'formality_variance', 'urgency_escalation',
    'reference_accuracy', 'entity_consistency', 'topic_coherence',
    'fabricated_history_score', 'relationship_velocity', 'cross_reference_count'
]

FEATURE_LABELS = [
    'Response Time\nVariance', 'Thread Length\n(Days)', 'Timestamp\nAnomalies',
    'Tone\nShift', 'Formality\nVariance', 'Urgency\nEscalation',
    'Reference\nAccuracy', 'Entity\nConsistency', 'Topic\nCoherence',
    'Fabricated\nHistory', 'Relationship\nVelocity', 'Cross-Reference\nCount'
]

FEATURE_CATEGORIES = {
    'Temporal':   ['response_time_variance', 'thread_length_days', 'timestamp_anomalies'],
    'Linguistic': ['tone_shift', 'formality_variance', 'urgency_escalation'],
    'Coherence':  ['reference_accuracy', 'entity_consistency', 'topic_coherence'],
    'Context':    ['fabricated_history_score', 'relationship_velocity', 'cross_reference_count'],
}

CATEGORY_COLORS = {
    'Temporal':   '#4C72B0',
    'Linguistic': '#DD8452',
    'Coherence':  '#55A868',
    'Context':    '#C44E52',
}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_all_features():
    """Load and combine train + val + test feature files into one DataFrame."""
    all_records = []
    for split in ['train', 'val', 'test']:
        records = json.load(open(f'../../data/features/{split}_features.json', encoding='utf-8'))
        for r in records:
            all_records.append(r)

    df = pd.DataFrame(all_records)
    df = df[FEATURE_COLS + ['label', 'attack_type']].copy()
    df[FEATURE_COLS] = df[FEATURE_COLS].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=FEATURE_COLS)
    return df


# ============================================================================
# CORRELATION COMPUTATION
# ============================================================================

def compute_correlation(df, cols=FEATURE_COLS):
    """Pearson correlation matrix + p-value matrix."""
    corr = df[cols].corr(method='pearson')

    n = len(cols)
    pval = pd.DataFrame(np.ones((n, n)), index=cols, columns=cols)
    for i in cols:
        for j in cols:
            if i != j:
                _, p = stats.pearsonr(df[i].dropna(), df[j].dropna())
                pval.loc[i, j] = p

    return corr, pval


def significance_mask(pval, alpha=0.05):
    """Return boolean mask — True where correlation is NOT significant."""
    return pval > alpha


# ============================================================================
# PLOTTING HELPERS
# ============================================================================

def category_color_bars(ax, feature_cols, position='left', bar_width=0.03):
    """Draw colored category bars along axes to group features visually."""
    fig = ax.get_figure()
    ax_pos = ax.get_position()

    for feat, color in zip(feature_cols, get_feature_colors(feature_cols)):
        idx = feature_cols.index(feat)
        n   = len(feature_cols)
        frac_start = 1 - (idx + 1) / n
        frac_end   = 1 - idx / n

        if position == 'left':
            rect = plt.Rectangle(
                (ax_pos.x0 - bar_width, ax_pos.y0 + frac_start * ax_pos.height),
                bar_width * 0.6,
                (frac_end - frac_start) * ax_pos.height,
                transform=fig.transFigure,
                color=color, clip_on=False
            )
        else:
            rect = plt.Rectangle(
                (ax_pos.x0 + idx / n * ax_pos.width,
                 ax_pos.y0 - bar_width),
                ax_pos.width / n,
                bar_width * 0.6,
                transform=fig.transFigure,
                color=color, clip_on=False
            )
        fig.add_artist(rect)


def get_feature_colors(feature_cols):
    colors = []
    for feat in feature_cols:
        for cat, feats in FEATURE_CATEGORIES.items():
            if feat in feats:
                colors.append(CATEGORY_COLORS[cat])
                break
    return colors


def make_heatmap(corr, pval, title, filename, annot=True, figsize=(10, 9)):
    """Generate a single annotated heatmap optimised for two-column LaTeX."""
    fig, ax = plt.subplots(figsize=figsize)

    short_labels = [
        'Response Time Var.', 'Thread Length Days', 'Timestamp Anomalies',
        'Tone Shift', 'Formality Variance', 'Urgency Escalation',
        'Reference Accuracy', 'Entity Consistency', 'Topic Coherence',
        'Fabricated History', 'Relationship Velocity', 'Cross-Ref Count',
    ]

    mask = significance_mask(pval)

    sns.heatmap(
        corr,
        ax=ax,
        annot=annot,
        fmt='.2f',
        cmap='RdYlBu_r',
        vmin=-1, vmax=1,
        center=0,
        square=True,
        linewidths=1.5,
        linecolor='white',
        mask=mask,
        annot_kws={'size': 9, 'weight': 'bold'},
        cbar_kws={'shrink': 0.75, 'label': 'Pearson r', 'pad': 0.02},
        xticklabels=short_labels,
        yticklabels=short_labels,
    )

    # Overlay non-significant cells in grey
    sns.heatmap(
        corr,
        ax=ax,
        annot=False,
        cmap=['#e8e8e8'],
        vmin=-1, vmax=1,
        square=True,
        linewidths=1.5,
        linecolor='white',
        mask=~mask,
        cbar=False,
        xticklabels=False,
        yticklabels=False,
    )

    # Re-apply tick labels (second heatmap call clears them)
    ax.set_xticks(np.arange(len(short_labels)) + 0.5)
    ax.set_yticks(np.arange(len(short_labels)) + 0.5)
    ax.set_xticklabels(short_labels, fontsize=12, fontweight='bold')
    ax.set_yticklabels(short_labels, fontsize=12, fontweight='bold')

    ax.set_title(title, fontsize=16, fontweight='bold', pad=18)

    ax.tick_params(axis='x', rotation=45, labelsize=12)
    ax.tick_params(axis='y', rotation=0,  labelsize=12)
    plt.setp(ax.get_xticklabels(), ha='right', rotation_mode='anchor')

    # Make colorbar label and ticks bold
    cbar = ax.collections[0].colorbar
    cbar.ax.set_ylabel('Pearson r', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=11)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight('bold')

    # Category legend
    legend_patches = [
        mpatches.Patch(color=color, label=cat)
        for cat, color in CATEGORY_COLORS.items()
    ]
    legend_patches.append(
        mpatches.Patch(color='#e8e8e8', label='Not significant (p>0.05)')
    )
    ax.legend(
        handles=legend_patches,
        title='Feature Category',
        loc='upper left',
        bbox_to_anchor=(1.22, 1.02),
        framealpha=0.95,
        fontsize=11,
        title_fontsize=12,
    )

    # Colour tick labels by category
    feat_colors = get_feature_colors(FEATURE_COLS)
    for tick, color in zip(ax.get_xticklabels(), feat_colors):
        tick.set_color(color)
        tick.set_fontweight('bold')
    for tick, color in zip(ax.get_yticklabels(), feat_colors):
        tick.set_color(color)
        tick.set_fontweight('bold')

    plt.subplots_adjust(left=0.22, bottom=0.22, right=0.72, top=0.92)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {filename}')


# ============================================================================
# FEATURE vs LABEL BAR CHART
# ============================================================================

def plot_feature_label_correlation(df, filename):
    """
    Compute Pearson r between each feature and the binary label (0/1).
    Shows which features best discriminate legitimate vs attack threads.
    """
    correlations = []
    pvalues      = []

    for feat in FEATURE_COLS:
        r, p = stats.pearsonr(df[feat], df['label'])
        correlations.append(r)
        pvalues.append(p)

    feat_df = pd.DataFrame({
        'feature':     FEATURE_COLS,
        'label_short': [l.replace('\n', ' ') for l in FEATURE_LABELS],
        'r':           correlations,
        'p':           pvalues,
        'significant': [p < 0.05 for p in pvalues],
        'color':       get_feature_colors(FEATURE_COLS),
    }).sort_values('r', key=abs, ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.barh(
        feat_df['label_short'],
        feat_df['r'],
        color=feat_df['color'],
        edgecolor='white',
        linewidth=0.8,
        alpha=0.88,
        height=0.65,
    )

    # Mark non-significant with hatching
    for bar, sig in zip(bars, feat_df['significant']):
        if not sig:
            bar.set_hatch('///')
            bar.set_alpha(0.4)

    ax.axvline(0,    color='black', linewidth=1.2)
    ax.axvline( 0.3, color='grey',  linewidth=0.8, linestyle='--', alpha=0.6)
    ax.axvline(-0.3, color='grey',  linewidth=0.8, linestyle='--', alpha=0.6)

    # Annotate r values
    for bar, r, p, sig in zip(bars, feat_df['r'], feat_df['p'], feat_df['significant']):
        x_pos = bar.get_width() + (0.015 if r >= 0 else -0.015)
        ha    = 'left' if r >= 0 else 'right'
        sig_marker = '' if sig else ' (ns)'
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f'r={r:.3f}{sig_marker}', va='center', ha=ha,
                fontsize=11, fontweight='bold')

    ax.set_xlabel('Pearson r with Label  (0 = Legitimate,  1 = Attack)',
                  fontsize=13, fontweight='bold', labelpad=8)
    ax.set_title('Feature–Label Correlation\n'
                 '(Which features best distinguish phishing from legitimate?)',
                 fontsize=15, fontweight='bold', pad=12)
    ax.set_xlim(-1.2, 1.35)

    # Bold y-tick labels coloured by category
    feat_label_colors = list(feat_df['color'])
    for tick, color in zip(ax.get_yticklabels(), feat_label_colors):
        tick.set_color(color)
        tick.set_fontweight('bold')

    legend_patches = [mpatches.Patch(color=c, label=cat) for cat, c in CATEGORY_COLORS.items()]
    legend_patches.append(
        mpatches.Patch(facecolor='grey', hatch='///', alpha=0.4, label='Not significant (p>0.05)')
    )
    ax.legend(
        handles=legend_patches,
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0),
        framealpha=0.95,
        fontsize=11,
        title_fontsize=12,
    )
    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(axis='x', labelsize=12)

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {filename}')

    return feat_df


# ============================================================================
# PRINT SUMMARY TABLE
# ============================================================================

def print_correlation_summary(corr, pval, title):
    print(f'\n--- {title} ---')
    print(f"{'Feature Pair':<55} {'r':>8} {'p-value':>12} {'Sig':>6}")
    print('-' * 85)

    pairs = []
    cols = FEATURE_COLS
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.loc[cols[i], cols[j]]
            p = pval.loc[cols[i], cols[j]]
            pairs.append((cols[i], cols[j], r, p))

    pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    for f1, f2, r, p in pairs[:15]:
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
        print(f"  {f1:<28} ↔ {f2:<24} {r:>8.4f} {p:>12.4e} {sig:>6}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print('=' * 65)
    print('PEARSON CORRELATION ANALYSIS — PHISHING DETECTION FEATURES')
    print('=' * 65)

    print('\n[1/5] Loading features...')
    df = load_all_features()
    df_legit  = df[df['label'] == 0]
    df_attack = df[df['label'] == 1]

    print(f'  Total threads : {len(df)}')
    print(f'  Legitimate    : {len(df_legit)}')
    print(f'  Attack        : {len(df_attack)}')

    print('\n[2/5] Computing correlation matrices...')
    corr_all,    pval_all    = compute_correlation(df)
    corr_legit,  pval_legit  = compute_correlation(df_legit)
    corr_attack, pval_attack = compute_correlation(df_attack)

    print('\n[3/5] Generating heatmaps...')

    make_heatmap(
        corr_all, pval_all,
        title='Pearson Correlation — All Threads (n={})'.format(len(df)),
        filename='../../figures/correlation_all.png',
    )
    make_heatmap(
        corr_legit, pval_legit,
        title='Pearson Correlation — Legitimate Threads Only (n={})'.format(len(df_legit)),
        filename='../../figures/correlation_legitimate.png',
    )
    make_heatmap(
        corr_attack, pval_attack,
        title='Pearson Correlation — Attack Threads Only (n={})'.format(len(df_attack)),
        filename='../../figures/correlation_attack.png',
    )

    print('\n[4/5] Generating feature-label correlation chart...')
    feat_label_df = plot_feature_label_correlation(df, '../../figures/correlation_feature_label.png')

    print('\n[5/5] Summary tables...')
    print_correlation_summary(corr_all, pval_all, 'Top 15 Feature Pairs by |r| — All Threads')

    print('\n--- Feature-Label Correlations (ranked by |r|) ---')
    print(f"{'Feature':<30} {'r':>8} {'p-value':>12} {'Sig':>6}")
    print('-' * 60)
    for _, row in feat_label_df.iterrows():
        sig = '***' if row['p'] < 0.001 else ('**' if row['p'] < 0.01 else ('*' if row['p'] < 0.05 else 'ns'))
        print(f"  {row['feature']:<28} {row['r']:>8.4f} {row['p']:>12.4e} {sig:>6}")

    print('\n' + '=' * 65)
    print('OUTPUT FILES')
    print('=' * 65)
    print('  correlation_all.png           — full dataset heatmap')
    print('  correlation_legitimate.png    — legitimate threads only')
    print('  correlation_attack.png        — attack threads only')
    print('  correlation_feature_label.png — which features predict attack label')

    print('\n' + '=' * 65)
    print('INTERPRETATION GUIDE')
    print('=' * 65)
    print('  r close to +1 : strong positive correlation')
    print('  r close to -1 : strong negative correlation')
    print('  r close to  0 : little/no linear relationship')
    print('  Grey cells    : correlation not significant (p > 0.05)')
    print('  |r| > 0.5     : strong correlation (worth noting)')
    print('  |r| 0.3-0.5   : moderate correlation')
    print('  |r| < 0.3     : weak correlation')


if __name__ == '__main__':
    main()
