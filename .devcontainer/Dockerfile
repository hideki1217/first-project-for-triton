FROM ubuntu:latest

RUN apt update && apt-get --no-install-recommends install -y \
    build-essential \
    git wget curl ca-certificates \
    python3-dev

WORKDIR /workspace

ARG UID=20000
ARG GID=20000
RUN set -eux; \
    # 既存ユーザー・グループを安全に削除
    if getent passwd "${UID}" >/dev/null; then \
        olduser=$(getent passwd "${UID}" | cut -d: -f1); \
        echo "UID ${UID} already used by ${olduser}, deleting..."; \
        userdel -r -f "${olduser}" || true; \
    fi; \
    if getent group "${GID}" >/dev/null; then \
        oldgroup=$(getent group "${GID}" | cut -d: -f1); \
        echo "GID ${GID} already used by ${oldgroup}, deleting..."; \
        groupdel "${oldgroup}" || true; \
    fi; \
    # 新規グループ・ユーザーを作成
    groupadd -g "${GID}" dev; \
    useradd -m -u "${UID}" -g "${GID}" -s /bin/bash dev; \
    # 念のためホーム権限を確認
    chown -R "${UID}:${GID}" /home/dev

USER dev

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
