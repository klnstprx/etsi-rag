package objectstore

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/s3"

	"etsi-rag/internal/config"
)

// opTimeout bounds each Put and Get, retries included, so a stalled store
// fails the call instead of hanging a crawl; it is far above a 10 MB body to
// a local RustFS.
const opTimeout = time.Minute

// Client wraps an S3-compatible client for object storage (RustFS).
type Client struct {
	s3      *s3.Client
	bucket  string
	timeout time.Duration // per Put or Get
}

// New creates a new object store client from config.
func New(cfg config.ObjectStoreConfig) (*Client, error) {
	if cfg.Endpoint == "" {
		return nil, fmt.Errorf("object store endpoint is required")
	}

	client := s3.New(s3.Options{
		BaseEndpoint: aws.String(cfg.Endpoint),
		Region:       cfg.Region,
		Credentials:  credentials.NewStaticCredentialsProvider(cfg.AccessKey, cfg.SecretKey, ""),
		UsePathStyle: true, // Required for MinIO/RustFS
	})

	return &Client{s3: client, bucket: cfg.Bucket, timeout: opTimeout}, nil
}

// Put stores data under the given key.
func (c *Client) Put(ctx context.Context, key string, data []byte, contentType string) error {
	ctx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()
	_, err := c.s3.PutObject(ctx, &s3.PutObjectInput{
		Bucket:      aws.String(c.bucket),
		Key:         aws.String(key),
		Body:        bytes.NewReader(data),
		ContentType: aws.String(contentType),
	})
	if err != nil {
		return fmt.Errorf("putting object %s: %w", key, err)
	}
	return nil
}

// Get retrieves data stored under the given key.
func (c *Client) Get(ctx context.Context, key string) ([]byte, error) {
	ctx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()
	resp, err := c.s3.GetObject(ctx, &s3.GetObjectInput{
		Bucket: aws.String(c.bucket),
		Key:    aws.String(key),
	})
	if err != nil {
		return nil, fmt.Errorf("getting object %s: %w", key, err)
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("reading object %s: %w", key, err)
	}
	return data, nil
}

// Healthy reports whether the bucket answers a HEAD request.
func (c *Client) Healthy(ctx context.Context) bool {
	_, err := c.s3.HeadBucket(ctx, &s3.HeadBucketInput{
		Bucket: aws.String(c.bucket),
	})
	return err == nil
}
