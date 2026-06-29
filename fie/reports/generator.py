"""
MatchReportGenerator — генератор PDF-отчёта матча.

Структура отчёта:
    Стр. 1  — Обложка (название, дата, длительность)
    Стр. 2  — Сводка матча (общая статистика)
    Стр. 3+ — Таблица рейтингов игроков
    Стр. N  — Физическая статистика топ-10 игроков (гистограммы)
    Стр. N  — Тепловые карты позиций (до 6 игроков на страницу)

Использование:
    from fie.reports.generator import MatchReportGenerator
    from fie.ratings.aggregator import MatchAggregator
    from fie.ratings.calculator import RatingCalculator, Position

    agg = MatchAggregator(fps=25)
    for frame in pipeline.process(source):
        agg.update(frame)

    calc = RatingCalculator(match_duration_seconds=agg.match_duration_seconds)
    ratings = calc.calculate_all(agg.get_all_stats())

    gen = MatchReportGenerator()
    pdf_path = gen.generate(
        output_path="match_report.pdf",
        ratings=ratings,
        aggregator=agg,
        match_title="Barcelona vs Real Madrid",
    )
"""

from __future__ import annotations

import io
import math
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from fie.ratings.aggregator import MatchAggregator, PhysicalStats, PositionalStats
from fie.ratings.calculator import PlayerRating, Position

# reportlab imports
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    Image,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.pdfgen import canvas as rl_canvas

# ---------------------------------------------------------------------------
# Цветовая схема
# ---------------------------------------------------------------------------

C_GREEN_DARK = colors.HexColor("#1a6b3a")
C_GREEN_MID = colors.HexColor("#2e8b57")
C_GREEN_LIGHT = colors.HexColor("#e8f5ed")
C_ACCENT = colors.HexColor("#f4a620")
C_DARK = colors.HexColor("#1c1c2e")
C_GRAY = colors.HexColor("#6b7280")
C_LIGHT_GRAY = colors.HexColor("#f3f4f6")
C_WHITE = colors.white
C_BLACK = colors.black


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_time(seconds: float) -> str:
    """Форматирует секунды в MM:SS."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _score_color(score: float) -> colors.HexColor:
    """Цвет рейтинга: зелёный (>=80), жёлтый (>=60), красный (<60)."""
    if score >= 80:
        return colors.HexColor("#16a34a")
    elif score >= 60:
        return colors.HexColor("#d97706")
    return colors.HexColor("#dc2626")


def _draw_heatmap_to_image(heatmap: np.ndarray, width_px: int = 200, height_px: int = 140) -> io.BytesIO:
    """
    Рисует тепловую карту 10×7 как PNG в памяти.
    Возвращает BytesIO с PNG.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, ax = plt.subplots(figsize=(width_px / 100, height_px / 100), dpi=100)
        ax.set_facecolor("#2d5a27")
        fig.patch.set_facecolor("#2d5a27")

        # Нормализовать
        hmax = heatmap.max()
        normed = heatmap / hmax if hmax > 0 else heatmap

        # Показать тепловую карту (транспонировать: X→горизонталь, Y→вертикаль)
        ax.imshow(
            normed.T,
            origin="lower",
            aspect="auto",
            cmap="RdYlGn",
            alpha=0.7,
            vmin=0,
            vmax=1,
            extent=[0, 10, 0, 7],
        )

        # Разметка поля
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 7)
        ax.axvline(x=3.33, color="white", alpha=0.4, linewidth=0.5)
        ax.axvline(x=6.66, color="white", alpha=0.4, linewidth=0.5)
        ax.axvline(x=5.0, color="white", alpha=0.2, linewidth=0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[:].set_visible(False)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.02,
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return buf
    except ImportError:
        # matplotlib не установлен — возвращаем пустой placeholder
        buf = io.BytesIO()
        return buf


# ---------------------------------------------------------------------------
# Page number canvas
# ---------------------------------------------------------------------------

class _NumberedCanvas(rl_canvas.Canvas):
    """Canvas с номерами страниц в подвале."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_pages: list = []

    def showPage(self):
        self._saved_pages.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self._saved_pages)
        for i, state in enumerate(self._saved_pages, 1):
            self.__dict__.update(state)
            self._draw_footer(i, total)
            super().showPage()
        super().save()

    def _draw_footer(self, page_num: int, total: int) -> None:
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(C_GRAY)
        self.drawRightString(
            A4[0] - 1.5 * cm,
            0.8 * cm,
            f"Page {page_num} / {total}",
        )
        self.drawString(
            1.5 * cm,
            0.8 * cm,
            "Football Intelligence Engine — Match Report",
        )
        self.setStrokeColor(C_LIGHT_GRAY)
        self.line(1.5 * cm, 1.2 * cm, A4[0] - 1.5 * cm, 1.2 * cm)
        self.restoreState()


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class MatchReportGenerator:
    """
    Генерирует PDF-отчёт матча из рейтингов и агрегированной статистики.

    Args:
        logo_path: Опциональный путь к PNG-логотипу (показывается на обложке).
    """

    PAGE_W, PAGE_H = A4
    MARGIN = 1.5 * cm

    def __init__(self, logo_path: str | Path | None = None) -> None:
        self._logo = Path(logo_path) if logo_path else None
        self._styles = getSampleStyleSheet()
        self._build_styles()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        output_path: str | Path,
        ratings: list[PlayerRating],
        aggregator: MatchAggregator,
        match_title: str = "Match Report",
        match_date: str | None = None,
        home_team: str = "Home",
        away_team: str = "Away",
    ) -> Path:
        """
        Создаёт PDF-файл и сохраняет по output_path.

        Returns:
            Path к созданному файлу.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        all_stats = aggregator.get_all_stats()
        duration = aggregator.match_duration_seconds
        match_date = match_date or datetime.now().strftime("%d %B %Y")

        # Документ
        doc = BaseDocTemplate(
            str(output_path),
            pagesize=A4,
            leftMargin=self.MARGIN,
            rightMargin=self.MARGIN,
            topMargin=self.MARGIN,
            bottomMargin=1.8 * cm,
        )

        # Фреймы
        content_frame = Frame(
            self.MARGIN,
            1.8 * cm,
            self.PAGE_W - 2 * self.MARGIN,
            self.PAGE_H - self.MARGIN - 1.8 * cm,
            id="content",
        )
        cover_frame = Frame(
            0, 0, self.PAGE_W, self.PAGE_H,
            id="cover",
        )

        doc.addPageTemplates([
            PageTemplate(id="cover_tpl", frames=[cover_frame]),
            PageTemplate(id="main", frames=[content_frame]),
        ])

        # Контент
        story = []
        story += self._build_cover(match_title, match_date, home_team, away_team, duration)
        story.append(NextPageTemplate("main"))
        story.append(PageBreak())
        story += self._build_summary(ratings, all_stats, duration)
        story.append(PageBreak())
        story += self._build_ratings_table(ratings)
        story.append(PageBreak())
        story += self._build_physical_stats(ratings, all_stats)

        # Тепловые карты (если matplotlib доступен)
        heatmap_pages = self._build_heatmaps(ratings, all_stats)
        if heatmap_pages:
            story.append(PageBreak())
            story += heatmap_pages

        doc.build(story, canvasmaker=_NumberedCanvas)
        return output_path

    # ------------------------------------------------------------------
    # Styles
    # ------------------------------------------------------------------

    def _build_styles(self) -> None:
        self._h1 = ParagraphStyle(
            "FIE_H1",
            fontName="Helvetica-Bold",
            fontSize=22,
            textColor=C_DARK,
            spaceAfter=8,
        )
        self._h2 = ParagraphStyle(
            "FIE_H2",
            fontName="Helvetica-Bold",
            fontSize=14,
            textColor=C_GREEN_DARK,
            spaceBefore=10,
            spaceAfter=6,
        )
        self._h3 = ParagraphStyle(
            "FIE_H3",
            fontName="Helvetica-Bold",
            fontSize=11,
            textColor=C_DARK,
            spaceBefore=6,
            spaceAfter=4,
        )
        self._body = ParagraphStyle(
            "FIE_Body",
            fontName="Helvetica",
            fontSize=9,
            textColor=C_DARK,
            spaceAfter=4,
        )
        self._small = ParagraphStyle(
            "FIE_Small",
            fontName="Helvetica",
            fontSize=8,
            textColor=C_GRAY,
        )
        self._cover_title = ParagraphStyle(
            "FIE_Cover_Title",
            fontName="Helvetica-Bold",
            fontSize=32,
            textColor=C_WHITE,
            alignment=1,
            spaceAfter=12,
        )
        self._cover_sub = ParagraphStyle(
            "FIE_Cover_Sub",
            fontName="Helvetica",
            fontSize=16,
            textColor=colors.HexColor("#d1fae5"),
            alignment=1,
            spaceAfter=6,
        )

    # ------------------------------------------------------------------
    # Pages
    # ------------------------------------------------------------------

    def _build_cover(
        self,
        title: str,
        date: str,
        home: str,
        away: str,
        duration: float,
    ) -> list:
        """Обложка с тёмным зелёным фоном."""

        class _CoverBackground:
            """Flowable, заливающий всю страницу зелёным."""
            def wrap(self, aw, ah):
                return (aw, ah)
            def draw(self):
                pass

        # Используем drawBackground через канвас в строках
        story = []

        # Spacer сверху
        story.append(Spacer(1, self.PAGE_H * 0.2))

        # Логотип / иконка
        story.append(Paragraph("⚽", ParagraphStyle(
            "icon", fontName="Helvetica", fontSize=48,
            textColor=C_WHITE, alignment=1,
        )))
        story.append(Spacer(1, 0.5 * cm))

        story.append(Paragraph("Football Intelligence Engine", ParagraphStyle(
            "fie_brand", fontName="Helvetica",
            fontSize=11, textColor=colors.HexColor("#86efac"),
            alignment=1, spaceAfter=2,
        )))
        story.append(Spacer(1, 0.3 * cm))

        story.append(Paragraph(title, self._cover_title))
        story.append(Paragraph(f"{home}  vs  {away}", self._cover_sub))
        story.append(Spacer(1, 0.5 * cm))

        story.append(Paragraph(date, ParagraphStyle(
            "cover_date", fontName="Helvetica",
            fontSize=13, textColor=colors.HexColor("#d1fae5"),
            alignment=1,
        )))
        story.append(Paragraph(
            f"Duration: {_fmt_time(duration)}",
            ParagraphStyle(
                "cover_dur", fontName="Helvetica",
                fontSize=11, textColor=colors.HexColor("#a7f3d0"),
                alignment=1,
            ),
        ))

        return story

    def _build_summary(
        self,
        ratings: list[PlayerRating],
        all_stats: dict[int, tuple[PhysicalStats, PositionalStats]],
        duration: float,
    ) -> list:
        """Страница со сводной статистикой матча."""
        story = []

        story.append(Paragraph("Match Summary", self._h1))
        story.append(HRFlowable(width="100%", thickness=2, color=C_GREEN_MID, spaceAfter=10))

        # Aggregate totals
        total_distance = sum(p.distance_total for p, _ in all_stats.values())
        total_sprints = sum(p.sprint_count for p, _ in all_stats.values())
        total_accel = sum(p.accel_high_count for p, _ in all_stats.values())
        max_speed = max((p.speed_max for p, _ in all_stats.values()), default=0)
        avg_rating = sum(r.overall for r in ratings) / len(ratings) if ratings else 0

        # Summary cards (3 per row)
        summary_data = [
            ("Players Tracked", str(len(all_stats))),
            ("Match Duration", _fmt_time(duration)),
            ("Avg Rating", f"{avg_rating:.1f}"),
            ("Total Distance", f"{total_distance / 1000:.1f} km"),
            ("Total Sprints", str(total_sprints)),
            ("Max Speed", f"{max_speed:.1f} km/h"),
            ("High Accels", str(total_accel)),
        ]

        # Таблица статистики
        card_data = []
        row = []
        for i, (label, val) in enumerate(summary_data):
            cell = Paragraph(
                f"<b><font size=18>{val}</font></b><br/><font size=8 color='#6b7280'>{label}</font>",
                ParagraphStyle("card", alignment=1, fontName="Helvetica"),
            )
            row.append(cell)
            if len(row) == 3 or i == len(summary_data) - 1:
                while len(row) < 3:
                    row.append("")
                card_data.append(row)
                row = []

        col_w = (self.PAGE_W - 2 * self.MARGIN) / 3
        tbl = Table(card_data, colWidths=[col_w] * 3, rowHeights=2 * cm)
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), C_GREEN_LIGHT),
            ("BOX", (0, 0), (-1, -1), 0.5, C_GREEN_MID),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, C_WHITE),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 0.5 * cm))

        # Топ-5 игроков
        story.append(Paragraph("Top 5 Players", self._h2))
        top5 = ratings[:5]
        if top5:
            t5_data = [["#", "Player ID", "Position", "Physical", "Tactical", "Overall"]]
            for i, r in enumerate(top5, 1):
                t5_data.append([
                    str(i),
                    f"Player {r.player_id}",
                    r.position.value,
                    f"{r.physical.overall:.1f}",
                    f"{r.tactical.overall:.1f}",
                    Paragraph(
                        f"<b><font color='{_score_color(r.overall).hexval()}'>{r.overall:.1f}</font></b>",
                        ParagraphStyle("sc", alignment=1),
                    ),
                ])
            cw = [(self.PAGE_W - 2 * self.MARGIN) / 6] * 6
            t5 = Table(t5_data, colWidths=cw)
            t5.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), C_GREEN_DARK),
                ("TEXTCOLOR", (0, 0), (-1, 0), C_WHITE),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_WHITE, C_GREEN_LIGHT]),
                ("GRID", (0, 0), (-1, -1), 0.3, C_LIGHT_GRAY),
                ("ROWPADDING", (0, 0), (-1, -1), 6),
            ]))
            story.append(t5)

        return story

    def _build_ratings_table(self, ratings: list[PlayerRating]) -> list:
        """Полная таблица рейтингов всех игроков."""
        story = []
        story.append(Paragraph("Player Ratings", self._h1))
        story.append(HRFlowable(width="100%", thickness=2, color=C_GREEN_MID, spaceAfter=10))

        header = ["#", "Player", "Pos", "Speed", "Accel", "Endur", "Intens",
                  "Phys", "Posit", "Press", "Cover", "Tact", "Overall"]
        table_data = [header]

        for i, r in enumerate(ratings, 1):
            overall_cell = Paragraph(
                f"<b><font color='{_score_color(r.overall).hexval()}'>{r.overall:.1f}</font></b>",
                ParagraphStyle("sc", alignment=1),
            )
            table_data.append([
                str(i),
                f"P{r.player_id}",
                r.position.value,
                f"{r.physical.speed:.0f}",
                f"{r.physical.acceleration:.0f}",
                f"{r.physical.endurance:.0f}",
                f"{r.physical.intensity:.0f}",
                f"{r.physical.overall:.1f}",
                f"{r.tactical.positioning:.0f}",
                f"{r.tactical.pressing:.0f}",
                f"{r.tactical.coverage:.0f}",
                f"{r.tactical.overall:.1f}",
                overall_cell,
            ])

        # Ширины колонок
        total_w = self.PAGE_W - 2 * self.MARGIN
        cw = [0.5 * cm, 1.1 * cm, 0.7 * cm] + [1.0 * cm] * 9 + [1.3 * cm]
        # Масштабируем под ширину страницы
        scale = total_w / sum(cw)
        cw = [w * scale for w in cw]

        tbl = Table(table_data, colWidths=cw, repeatRows=1)
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), C_GREEN_DARK),
            ("TEXTCOLOR", (0, 0), (-1, 0), C_WHITE),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7.5),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_WHITE, C_GREEN_LIGHT]),
            ("GRID", (0, 0), (-1, -1), 0.3, C_LIGHT_GRAY),
            ("ROWPADDING", (0, 0), (-1, -1), 4),
            # Выделить Overall жирным
            ("FONTNAME", (-1, 1), (-1, -1), "Helvetica-Bold"),
        ]))
        story.append(tbl)

        story.append(Spacer(1, 0.3 * cm))
        story.append(Paragraph(
            "Speed/Accel/Endur/Intens = Physical sub-scores | Posit/Press/Cover = Tactical sub-scores | "
            "Phys/Tact = category overall | Overall = weighted final score",
            self._small,
        ))

        return story

    def _build_physical_stats(
        self,
        ratings: list[PlayerRating],
        all_stats: dict[int, tuple[PhysicalStats, PositionalStats]],
    ) -> list:
        """Детальная физическая статистика топ-10 игроков."""
        story = []
        story.append(Paragraph("Physical Statistics — Top 10", self._h1))
        story.append(HRFlowable(width="100%", thickness=2, color=C_GREEN_MID, spaceAfter=10))

        top10_ids = [r.player_id for r in ratings[:10]]
        rows = [["Player", "Distance (m)", "Walk (m)", "Run (m)", "Sprint (m)",
                 "Max Speed", "Avg Speed", "Sprints", "Hi-Accel", "Ball Time"]]
        for pid in top10_ids:
            if pid not in all_stats:
                continue
            p, _ = all_stats[pid]
            rows.append([
                f"Player {pid}",
                f"{p.distance_total:.0f}",
                f"{p.distance_walk:.0f}",
                f"{p.distance_run + p.distance_high_run:.0f}",
                f"{p.distance_sprint:.0f}",
                f"{p.speed_max:.1f}",
                f"{p.speed_avg:.1f}",
                str(p.sprint_count),
                str(p.accel_high_count),
                f"{p.time_near_ball:.1f}s",
            ])

        total_w = self.PAGE_W - 2 * self.MARGIN
        cw = [1.5 * cm] + [1.5 * cm] * 9
        scale = total_w / sum(cw)
        cw = [w * scale for w in cw]

        tbl = Table(rows, colWidths=cw, repeatRows=1)
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), C_GREEN_DARK),
            ("TEXTCOLOR", (0, 0), (-1, 0), C_WHITE),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("ALIGN", (0, 0), (0, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_WHITE, C_GREEN_LIGHT]),
            ("GRID", (0, 0), (-1, -1), 0.3, C_LIGHT_GRAY),
            ("ROWPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(tbl)

        # Speed zones breakdown
        story.append(Spacer(1, 0.5 * cm))
        story.append(Paragraph("Speed Zones Distribution — Top 5", self._h2))

        zone_rows = [["Player", "Walk\n(<7)", "Jog\n(7-14)", "Run\n(14-21)", "Hi-Run\n(21-25)", "Sprint\n(>25)"]]
        for pid in top10_ids[:5]:
            if pid not in all_stats:
                continue
            p, _ = all_stats[pid]
            total = max(p.distance_total, 1)
            zone_rows.append([
                f"Player {pid}",
                f"{p.distance_walk:.0f}m ({p.distance_walk/total*100:.0f}%)",
                f"{p.distance_jog:.0f}m ({p.distance_jog/total*100:.0f}%)",
                f"{p.distance_run:.0f}m ({p.distance_run/total*100:.0f}%)",
                f"{p.distance_high_run:.0f}m ({p.distance_high_run/total*100:.0f}%)",
                f"{p.distance_sprint:.0f}m ({p.distance_sprint/total*100:.0f}%)",
            ])

        col_w2 = total_w / 6
        tbl2 = Table(zone_rows, colWidths=[col_w2] * 6, repeatRows=1)
        tbl2.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), C_GREEN_MID),
            ("TEXTCOLOR", (0, 0), (-1, 0), C_WHITE),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_WHITE, C_GREEN_LIGHT]),
            # Цвет колонок по зонам
            ("BACKGROUND", (1, 1), (1, -1), colors.HexColor("#dcfce7")),
            ("BACKGROUND", (2, 1), (2, -1), colors.HexColor("#bbf7d0")),
            ("BACKGROUND", (3, 1), (3, -1), colors.HexColor("#fef9c3")),
            ("BACKGROUND", (4, 1), (4, -1), colors.HexColor("#fed7aa")),
            ("BACKGROUND", (5, 1), (5, -1), colors.HexColor("#fecaca")),
            ("GRID", (0, 0), (-1, -1), 0.3, C_LIGHT_GRAY),
            ("ROWPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(tbl2)

        return story

    def _build_heatmaps(
        self,
        ratings: list[PlayerRating],
        all_stats: dict[int, tuple[PhysicalStats, PositionalStats]],
    ) -> list:
        """Тепловые карты позиций для топ-12 игроков (3 строки × 4 колонки)."""
        try:
            import matplotlib  # noqa
        except ImportError:
            return []

        story = []
        story.append(Paragraph("Position Heatmaps — Top 12", self._h1))
        story.append(HRFlowable(width="100%", thickness=2, color=C_GREEN_MID, spaceAfter=8))
        story.append(Paragraph(
            "X axis = field length (0=own goal, 105=opponent goal) | "
            "Y axis = field width | Darker green = more time spent",
            self._small,
        ))
        story.append(Spacer(1, 0.3 * cm))

        total_w = self.PAGE_W - 2 * self.MARGIN
        cols = 4
        cell_w = total_w / cols
        img_w = cell_w - 0.4 * cm
        img_h = img_w * 0.65

        top12 = ratings[:12]
        row_data: list = []
        cur_row: list = []

        for r in top12:
            pid = r.player_id
            if pid not in all_stats:
                cur_row.append("")
                if len(cur_row) == cols:
                    row_data.append(cur_row)
                    cur_row = []
                continue

            _, pos_stats = all_stats[pid]
            buf = _draw_heatmap_to_image(pos_stats.heatmap,
                                         width_px=int(img_w * 50),
                                         height_px=int(img_h * 50))
            if buf.getbuffer().nbytes == 0:
                cur_row.append("")
            else:
                img = Image(buf, width=img_w, height=img_h)
                label = Paragraph(
                    f"<b>Player {pid}</b> | {r.position.value} | {r.overall:.1f}",
                    ParagraphStyle("hm_label", fontName="Helvetica", fontSize=7,
                                   alignment=1, textColor=C_DARK),
                )
                cur_row.append([img, label])

            if len(cur_row) == cols:
                row_data.append(cur_row)
                cur_row = []

        if cur_row:
            while len(cur_row) < cols:
                cur_row.append("")
            row_data.append(cur_row)

        if not row_data:
            return []

        # Собрать в таблицу (img + label в каждой ячейке)
        # Конвертируем пары [img, label] во вложенные таблицы
        tbl_rows = []
        for row in row_data:
            tbl_row = []
            for cell in row:
                if isinstance(cell, list):
                    inner = Table([[cell[0]], [cell[1]]], colWidths=[img_w])
                    inner.setStyle(TableStyle([
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("TOPPADDING", (0, 0), (-1, -1), 2),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                    ]))
                    tbl_row.append(inner)
                else:
                    tbl_row.append("")
            tbl_rows.append(tbl_row)

        hm_table = Table(tbl_rows, colWidths=[cell_w] * cols)
        hm_table.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.3, C_LIGHT_GRAY),
            ("ROWPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(hm_table)

        return story
