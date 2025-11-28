import base64
from io import BytesIO

from curl_cffi import AsyncSession, ProxySpec, requests
from curl_cffi.requests.exceptions import Timeout
from PIL import Image

from astrbot.api import logger


class Utils:
    def __init__(
        self,
        network_config: dict,
        def_params: dict,
        max_retry: int,
    ):
        # 初始化异步HTTP会话
        proxy = network_config.get("proxy", None)
        self.proxies: ProxySpec | None = (
            {"http": proxy, "https": proxy} if proxy else None
        )
        self.session = AsyncSession(
            impersonate="chrome136", timeout=network_config.get("timeout", 600)
        )

        # 默认参数
        self.image_size = def_params.get("image_size", "1K")
        self.aspect_ratio = def_params.get("aspect_ratio", "default")
        self.google_search = def_params.get("google_search", False)
        self.text_response = def_params.get("text_response", False)
        self.max_retry = max_retry

    def _handle_image(self, image_bytes: bytes) -> tuple[str, str]:
        try:
            with Image.open(BytesIO(image_bytes)) as img:
                fmt = (img.format or "").upper()
                # 如果不是 GIF，直接返回原图
                if fmt != "GIF":
                    mime = f"image/{fmt.lower()}"
                    b64 = base64.b64encode(image_bytes).decode("utf-8")
                    return (mime, b64)
                # 处理 GIF
                buf = BytesIO()
                # 取第一帧
                img.seek(0)
                img = img.convert("RGBA")
                img.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                return ("image/png", b64)
        except Exception as e:
            logger.warning(f"GIF 处理失败，返回原图: {e}")
            b64 = base64.b64encode(image_bytes).decode("utf-8")
            return ("image/gif", b64)

    async def _download_image(
        self, url: str
    ) -> tuple[tuple[str, str] | None, str | None]:
        try:
            response = await self.session.get(url)
            content = self._handle_image(response.content)
            return content, None
        except (
            requests.exceptions.SSLError,
            requests.exceptions.CertificateVerifyError,
        ):
            # 关闭SSL验证
            response = await self.session.get(url, verify=False)
            content = self._handle_image(response.content)
            return content, None
        except Timeout as e:
            logger.error(f"网络请求超时: {e}")
            return None, "下载图片失败：请求超时"
        except Exception as e:
            logger.error(f"下载图片失败: {e}")
            return None, "下载图片失败"

    async def fetch_images(self, image_urls: list[str]) -> list[tuple[str, str]]:
        image_b64_list = []
        for url in image_urls:
            content, error = await self._download_image(url)
            if error or content is None:
                continue
            image_b64_list.append(content)
        return image_b64_list

    def _build_gemini_context(
        self,
        model: str,
        prompt: str,
        image_b64_list: list[tuple[str, str]],
        params: dict,
    ) -> dict:
        # 处理图片内容部分
        parts = []
        for mime, b64 in image_b64_list:
            parts.append(
                {
                    "inlineData": {
                        "mimeType": mime,
                        "data": b64,
                    }
                }
            )

        # 处理响应内容的类型
        responseModalities = ["IMAGE"]
        if params.get("text_response", self.text_response):
            responseModalities.insert(0, "TEXT")

        # 构建请求上下文
        context = {
            "contents": [{"parts": [{"text": prompt}, *parts]}],
            "generationConfig": {
                "responseModalities": responseModalities,
            },
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF"},
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "OFF",
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "OFF",
                },
            ],
        }

        # 处理图片宽高比参数
        aspect_ratio = params.get("aspect_ratio", self.aspect_ratio)
        if aspect_ratio != "default":
            context["generationConfig"]["imageConfig"] = {"aspectRatio": aspect_ratio}

        # 以下参数仅 Gemini-3-Pro-Image-Preview 模型有效
        if model == "gemini-3-pro-image-preview":
            # 处理工具类
            if (
                params.get("google_search", self.google_search)
                and model == "gemini-3-pro-image-preview"
            ):
                context["tools"] = [{"google_search": {}}]
            # 处理图片分辨率参数
            image_size = params.get("image_size", self.image_size)
            context["generationConfig"]["imageConfig"] = {"imageSize": image_size}

        return context

    async def _call_gemini_api(
        self,
        api_url: str,
        api_key: str,
        model: str,
        prompt: str,
        image_b64_list: list[tuple[str, str]],
        params: dict,
    ) -> tuple[list[tuple[str, str]] | None, str | None]:
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        }
        url = f"{api_url}/{model}:generateContent"
        gemini_context = self._build_gemini_context(
            model, prompt, image_b64_list, params
        )
        try:
            response = await self.session.post(
                url, headers=headers, json=gemini_context, proxies=self.proxies
            )
            result = response.json()
            if response.status_code == 200:
                b64_images = []
                for item in result.get("candidates", []):
                    # 检查 finishReason 状态
                    finishReason = item.get("finishReason", "")
                    if finishReason == "STOP":
                        parts = item.get("content", {}).get("parts", [])
                        for part in parts:
                            if "inlineData" in part and "data" in part["inlineData"]:
                                data = part["inlineData"]
                                b64_images.append((data["mimeType"], data["data"]))
                    else:
                        logger.warning(f"图片生成失败, 响应内容: {response.text}")
                        return None, f"图片生成失败，原因: {finishReason}"
                if not b64_images:
                    logger.warning(
                        f"请求成功，但未返回图片数据, 响应内容: {response.text}"
                    )
                    if result.get("promptFeedback", {}):
                        return (
                            None,
                            f"请求被内容安全系统拦截，原因：{result.get('promptFeedback', {}).get('blockReason', '未获取到原因')}",
                        )
                    return None, "响应中未包含图片数据"
                return b64_images, None
            else:
                logger.error(
                    f"图片生成失败，状态码: {response.status_code}, 响应内容: {response.text}"
                )
                err_msg = result.get("error", {}).get("message", "未知原因")
                return None, f"图片生成失败：{err_msg}"
        except Timeout as e:
            logger.error(f"网络请求超时: {e}")
            return None, "图片生成失败：响应超时"
        except Exception as e:
            logger.error(f"请求错误: {e}")
            return None, "图片生成失败"

    def _build_openai_context(
        self, model, prompt: str, image_b64_list: list[tuple[str, str]]
    ) -> dict:
        images_content = []
        for mime, b64 in image_b64_list:
            images_content.append(
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
            )
        context = {
            "model": model,
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}, *images_content],
                }
            ],
        }
        return context

    async def _call_openai_api(
        self,
        api_url: str,
        model,
        api_key: str,
        prompt: str,
        image_b64_list: list[tuple[str, str]],
    ):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        openai_context = self._build_openai_context(model, prompt, image_b64_list)
        try:
            response = await self.session.post(
                api_url, headers=headers, json=openai_context, proxies=self.proxies
            )
            result = response.json()
            if response.status_code == 200:
                pass
            else:
                logger.error(
                    f"图片生成失败，状态码: {response.status_code}, 响应内容: {response.text}"
                )
                return None, result.get("error", {}).get("message", "未知原因")
        except Timeout as e:
            logger.error(f"网络请求超时: {e}")
            return None, "图片生成失败：响应超时"
        except Exception as e:
            logger.error(f"请求错误: {e}")
            return None, "图片生成失败"

    async def generate_images(
        self,
        api_type: str,
        api_url: str,
        model: str,
        api_key: str,
        prompt: str,
        image_b64_list: list[tuple[str, str]] = [],
        params: dict = {},
    ) -> tuple[list[tuple[str, str]] | None, str | None]:
        for _ in range(self.max_retry):
            if api_type == "Gemini":
                image_b64, err = await self._call_gemini_api(
                    api_url=api_url,
                    model=model,
                    api_key=api_key,
                    prompt=prompt,
                    image_b64_list=image_b64_list,
                    params=params,
                )
            else:
                # 不同平台的OpenAI规范接口返回的响应不太统一，留待以后再做。
                logger.error(f"不支持的API类型: {api_type}")
                return None, "❌ 不支持的API类型"
            if err is None:
                return image_b64, None
            logger.warning(f"图片生成失败，当前Key重试次数: {_ + 1}")
        return None, f"❌ {err}"

    async def close(self):
        await self.session.close()
