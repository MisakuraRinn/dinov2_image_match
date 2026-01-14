import json
import mimetypes
from pathlib import Path
from urllib.parse import quote

from django.conf import settings
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.core.files.storage import FileSystemStorage
from django.http import FileResponse, Http404
from django.shortcuts import redirect, render
from django.views.decorators.http import require_http_methods

from .models import SearchHistory
from .retriever import get_retriever


def _apply_form_classes(form):
    for field in form.fields.values():
        existing = field.widget.attrs.get("class", "")
        field.widget.attrs["class"] = (existing + " input").strip()


def _attach_image_urls(results):
    for item in results:
        relpath = item.get("img_path", "")
        if relpath:
            safe_path = relpath.replace("\\", "/")
            item["image_url"] = f"/gallery/{quote(safe_path, safe='/')}/"
        else:
            item["image_url"] = ""


def home(request):
    return render(request, "retrieval/home.html")


def search(request):
    if request.method == "GET":
        return render(request, "retrieval/index.html", {"default_k": 10})

    if "query_image" not in request.FILES:
        return render(request, "retrieval/index.html", {"error": "Please upload an image.", "default_k": 10})

    upload = request.FILES["query_image"]
    try:
        top_k = int(request.POST.get("top_k", "10"))
    except ValueError:
        top_k = 10
    top_k = max(1, min(top_k, 100))

    fs = FileSystemStorage(location=settings.MEDIA_ROOT / "uploads")
    filename = fs.save(upload.name, upload)
    upload_path = Path(fs.path(filename))
    query_relpath = f"uploads/{filename}"
    query_url = f"{settings.MEDIA_URL}{query_relpath}"

    retriever = get_retriever()
    if retriever is None:
        return render(request, "retrieval/index.html", {
            "error": "Feature bank or weights not found. Check feature_bank.npz and vit-dinov2-base.npz.",
            "default_k": 10,
        })

    base_results = retriever.search(str(upload_path), top_k)
    results = [dict(item) for item in base_results]
    _attach_image_urls(results)

    saved_to_history = False
    if request.user.is_authenticated:
        SearchHistory.objects.create(
            user=request.user,
            query_image=query_relpath,
            top_k=top_k,
            results_json=json.dumps(base_results, ensure_ascii=False),
        )
        saved_to_history = True

    context = {
        "query_url": query_url,
        "results": results,
        "top_k": top_k,
        "saved_to_history": saved_to_history,
    }
    return render(request, "retrieval/results.html", context)


@login_required
def history(request):
    entries = SearchHistory.objects.filter(user=request.user).order_by("-created_at")
    items = []
    for entry in entries:
        results = json.loads(entry.results_json)
        _attach_image_urls(results)
        items.append({
            "id": entry.id,
            "created_at": entry.created_at,
            "query_url": f"{settings.MEDIA_URL}{entry.query_image}",
            "top_k": entry.top_k,
            "results": results,
        })
    return render(request, "retrieval/history.html", {"items": items})


@login_required
@require_http_methods(["POST"])
def delete_history(request, entry_id):
    SearchHistory.objects.filter(id=entry_id, user=request.user).delete()
    return redirect("retrieval:history")


def login_view(request):
    if request.user.is_authenticated:
        return redirect("retrieval:search")

    form = AuthenticationForm(request, data=request.POST or None)
    _apply_form_classes(form)

    if request.method == "POST" and form.is_valid():
        login(request, form.get_user())
        return redirect("retrieval:search")

    return render(request, "retrieval/login.html", {"form": form})


def register(request):
    if request.user.is_authenticated:
        return redirect("retrieval:search")

    form = UserCreationForm(request.POST or None)
    _apply_form_classes(form)

    if request.method == "POST" and form.is_valid():
        user = form.save()
        login(request, user)
        return redirect("retrieval:search")

    return render(request, "retrieval/register.html", {"form": form})


def logout_view(request):
    logout(request)
    return redirect("retrieval:home")


def gallery_image(request, relpath):
    data_root = Path(settings.DATA_ROOT).resolve()
    full_path = (data_root / relpath).resolve()
    try:
        full_path.relative_to(data_root)
    except ValueError:
        raise Http404("Invalid path")
    if not full_path.exists() or not full_path.is_file():
        raise Http404("Image not found")

    content_type, _ = mimetypes.guess_type(str(full_path))
    response = FileResponse(full_path.open("rb"), content_type=content_type or "application/octet-stream")
    return response
